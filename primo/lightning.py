import lightning as L
import torch
from .modules.set_transformer import ProteinSetTransformer

from torchmetrics import MeanSquaredError
from torchmetrics.regression import SpearmanCorrCoef
from .utils.metrics import AverageSpearmanCorrelation
from typing import Iterable
import math

class LightningSetTransformer(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = ProteinSetTransformer(
            num_layers = cfg.model.num_layers,
            embedding_size=cfg.model.embedding_size,
            hidden_size=cfg.model.hidden_size,
            num_attention_heads=cfg.model.num_attention_heads,
            ff_factor=cfg.model.ff_factor,
            dropout=cfg.model.dropout,
            num_labels = cfg.data.num_labels,
            num_auxiliary_labels=len(cfg.data.auxiliary_labels),
            target_predict_mode=cfg.model.target_predict_mode,
            loss_fn=cfg.model.loss_fn,
            attn_method=cfg.model.attn_method,
            auxiliary_label_mode=cfg.model.auxiliary_label_mode,
        )

        self.log_debug = False
        self.enable_mc_dropout = False
        self.mc_dropout_samples =1

        if cfg.training.compile:
            self.model = torch.compile(self.model)


        # make an MSE for each label type
        self.training_metric = cfg.training.metric
        if cfg.training.metric == "mse":
            self.mses = torch.nn.ModuleDict() # torchmetrics can be on device
            for i in range(cfg.data.num_labels):
                self.mses[f"train_label_{i}"] = MeanSquaredError()
                self.mses[f"train_label_{i}_nomask"] = MeanSquaredError()
                self.mses[f"val_label_{i}"] = MeanSquaredError()
                self.mses[f"val_label_{i}_nomask"] = MeanSquaredError()
            self.train_micro_mse = MeanSquaredError()
            self.val_micro_mse = MeanSquaredError()
        elif cfg.training.metric == "spearman":
            self.spearmans = torch.nn.ModuleDict()
            for i in range(cfg.data.num_labels):
                self.spearmans[f"train_label_{i}"] = AverageSpearmanCorrelation(mask_idx=-100)
                self.spearmans[f"val_label_{i}"] = AverageSpearmanCorrelation(mask_idx=-100)
            self.train_micro_spearman = AverageSpearmanCorrelation(mask_idx=-100)


        self.test_spearman = SpearmanCorrCoef()


        bsize = self.cfg.training.batch_size * self.cfg.training.grad_accumulation_steps
        self.total_steps = (self.cfg.training.epochs * self.cfg.data.sets_per_epoch/bsize)


    def configure_optimizers(self):
        # params = self.model.parameters()
        # optimizer = torch.optim.AdamW(params, lr=self.cfg.training.lr, weight_decay=0.01)

        grouped_params = [
        {
            "params": [p for n, p in self.model.named_parameters() if not 'auxiliary_mixing_alpha' in n],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in self.model.named_parameters() if 'auxiliary_mixing_alpha' in n],
            "weight_decay": 0.0, # TODO controllable
        }

        ]
        optimizer = torch.optim.AdamW(grouped_params, lr=self.cfg.training.lr)

        if  hasattr(self.cfg, "is_finetuning") and self.cfg.is_finetuning:
            # we don't want to schedule the learning rate for the finetuning
            return optimizer

        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer,[
            # linear warmup for cfg.training.warmup_steps
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.0001, end_factor=1, total_iters=self.cfg.training.warmup_steps),
            # linear decay to zero
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0, total_iters=self.total_steps - self.cfg.training.warmup_steps +1)
        ],
        [self.cfg.training.warmup_steps] # the milestones for the scheduler
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            }
        }
    


    def _log_metrics_with_mask(self, label_ids, labels_in, label_preds, labels, prefix='train'):
        """Helper function to log class-wise metrics and ignore samples where the label wasn't masked."""


        if self.training_metric == "mse":

            # flatten everything
            label_ids_flat = label_ids.flatten()
            labels_flat = labels.flatten()
            label_preds_flat = label_preds.flatten()
            labels_in_flat = labels_in.flatten()


            for i in range(self.cfg.data.num_labels):
                # NOTE we are only interested in the MSE for labels that were actually masked in `labels` - 2 is the mask token.
                selected = label_ids_flat == i
                was_masked = labels_in_flat[selected] == self.cfg.training.masker.label_mask_token_id

                self.mses[f"{prefix}_label_{i}"].update(label_preds_flat[selected][was_masked], labels_flat[selected][was_masked])
                getattr(self, f"{prefix}_micro_mse").update(label_preds_flat[selected][was_masked], labels_flat[selected][was_masked])

                # log the MSE for the non-masked samples
                if self.log_debug:
                    self.mses[f"{prefix}_label_{i}_nomask"].update(label_preds_flat[selected][~was_masked], labels_flat[selected][~was_masked])

        elif self.training_metric == "spearman":
            
            # set nomask targets to -100
            targets = labels.clone() # b, n
            targets[labels_in != self.cfg.training.masker.label_mask_token_id] = -100

            for i in range(self.cfg.data.num_labels):
                selected = label_ids == i # b, n
                self.spearmans[f"{prefix}_label_{i}"].update(
                    label_preds[selected].view(-1, label_preds.size(1)), 
                    targets[selected].view(-1, targets.size(1))
                                           )
            
            self.train_micro_spearman.update(label_preds, targets)


    def _get_reconstruction_loss_coefficient(self) -> float:
        """Helper function to get the coefficient for the reconstruction loss."""
        # NOTE lighning's global_step does account for the gradient accumulation steps. Will only update every 
        # gradient_accumulation_steps batches.

        ratio_total_steps = self.global_step / self.total_steps
        cosine_scaler = 0.5 * (1.0 + math.cos(math.pi * ratio_total_steps))
        reconstruction_loss_coeff = self.cfg.training.end_mlm_coefficient + cosine_scaler * (self.cfg.training.start_mlm_coefficient - self.cfg.training.end_mlm_coefficient)
        return reconstruction_loss_coeff
    



    def training_step(self, batch, batch_idx):
        

        masked_tokens, pad_mask, label_ids, labels, auxiliary_labels, seq_tokens, labels_target = batch

        protein_logits, label_preds, losses = self.model(
            masked_tokens, label_ids, labels, auxiliary_labels, pad_mask, seq_tokens, labels_target
        )


        mlm_coefficient = self._get_reconstruction_loss_coefficient()
        total_loss = losses["label"] + mlm_coefficient * losses["protein"] 
        self.log("train_mlm_loss", losses["protein"])
        self.log("train_label_loss", losses["label"])
        self.log("train_total_loss", total_loss)

        self._log_metrics_with_mask(label_ids, labels, label_preds, labels_target, prefix='train')


        # log schedulers to be sure they are doing what we expect
        self.log("scheduling/lr", self.trainer.optimizers[0].param_groups[0]['lr'])
        self.log("scheduling/mlm_coefficient", mlm_coefficient)


        if self.model.auxiliary_label_mode in ['add', 'embed_and_add']:
            self.log("auxiliary_mixing_alpha", self.model.auxiliary_mixing_alpha.item())


        return total_loss
    

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        

        masked_tokens, pad_mask, label_ids, labels, auxiliary_labels, seq_tokens, label_target = batch

        protein_logits, label_logits, losses = self.model(
            masked_tokens, label_ids, labels, auxiliary_labels, pad_mask, seq_tokens, label_target
        )

        mlm_coefficient = self._get_reconstruction_loss_coefficient()
        total_loss = losses["label"] + mlm_coefficient * losses["protein"]

        self.log("val_mlm_loss", losses["protein"])
        self.log("val_label_loss", losses["label"])
        self.log("val_total_loss", total_loss)

        self._log_metrics_with_mask(label_ids, labels, label_logits, label_target, prefix='val')


        return None
    
    def on_train_epoch_end(self):

        if self.training_metric == 'mse':
        
            mses = []
            for i in range(self.cfg.data.num_labels):
                mse = self.mses[f"train_label_{i}"].compute()
                self.log(f"properties/train_label_{i}_mse", mse)

                self.mses[f"train_label_{i}"].reset()
                mses.append(mse)

            self.log("train_macro_mse", sum(mses) / len(mses))
            self.log("train_micro_mse", self.train_micro_mse.compute())
            self.train_micro_mse.reset()
        elif self.training_metric == 'spearman':
            for i in range(self.cfg.data.num_labels):
                self.log(f"properties/train_label_{i}_spearman", self.spearmans[f"train_label_{i}"].compute())
                self.spearmans[f"train_label_{i}"].reset()

            self.log("train_micro_spearman", self.train_micro_spearman.compute())
            self.train_micro_spearman.reset()


    def on_validation_epoch_end(self):


        if self.training_metric == 'mse':

            mses = []

            for i in range(self.cfg.data.num_labels):
                mse = self.mses[f"val_label_{i}"].compute()
                self.log(f"properties/val_label_{i}_mse", mse)

                self.mses[f"val_label_{i}"].reset()
                mses.append(mse)

            self.log('val_macro_mse', sum(mses) / len(mses))
            self.log("val_micro_mse", self.val_micro_mse.compute())
            self.val_micro_mse.reset()

        elif self.training_metric == 'spearman':
            for i in range(self.cfg.data.num_labels):
                self.log(f"properties/val_label_{i}_spearman", self.spearmans[f"val_label_{i}"].compute())
                self.spearmans[f"val_label_{i}"].reset()

            self.log("val_micro_spearman", self.train_micro_spearman.compute())
            self.train_micro_spearman.reset()


    def predict_step(self, batch, batch_idx, dataloader_idx=0):

        if self.enable_mc_dropout:
            self.model.train()
            
        masked_tokens, pad_mask, label_ids, labels, auxiliary_labels, seq_tokens, label_target = batch

        all_preds = []
        for _ in range(self.mc_dropout_samples):
            protein_logits, label_logits, _ = self.model(
                masked_tokens, label_ids, labels, auxiliary_labels, pad_mask, seq_tokens, label_target
            )
            all_preds.append(label_logits[:, -1]) # (b, n) -> (b)

        label_logits = torch.stack(all_preds, dim=-1) # (b, mc_samples)

        # protein_logits, label_logits, _ = self.model(
        #     masked_tokens, label_ids, labels, auxiliary_labels, pad_mask, seq_tokens, None, use_cache=True
        # )

        # label_logits = label_logits[:, -1] # (b_size, set_size) -> (b_size)
        # label_target = label_target[:, -1]
        
        return label_logits
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        masked_tokens, pad_mask, label_ids, labels, auxiliary_labels, seq_tokens, label_target = batch

        protein_logits, label_logits, _ = self.model(
            masked_tokens, label_ids, labels, auxiliary_labels, pad_mask, seq_tokens, None, use_cache=True
        )

        # # Get -1 index in the set dimension
        label_logits = label_logits[:, -1]
        label_target = label_target[:, -1]

        self.test_spearman.update(label_logits, label_target)

        return label_logits
    
    def on_test_epoch_end(self):
        spearman = self.test_spearman.compute()
        self.log("test_spearman", spearman)
        self.test_spearman.reset()



    # def predict_step(self, batch, batch_idx, dataloader_idx=0):
    #     protein_1_tokens_masked, protein_2_tokens_masked, pad_mask_1, pad_mask_2, labels, auxiliary_labels, protein_1_tokens, protein_2_tokens = batch

    #     protein_1_logits, protein_2_logits, label_logits, _ = self.model(
    #         protein_1_tokens_masked, protein_2_tokens_masked, labels, auxiliary_labels, pad_mask_1, pad_mask_2
    #     )

    #     return label_logits


