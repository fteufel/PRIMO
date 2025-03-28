import lightning as L
import torch
from .modules.baselines import AugmentedPropertyPredictor, MLPPropertyPredictor

from torchmetrics import MeanSquaredError
from torchmetrics.regression import SpearmanCorrCoef
from .utils.metrics import AverageSpearmanCorrelation

class LightningAugmentedRidgeRegression(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = AugmentedPropertyPredictor(
            embedding_size=cfg.model.embedding_size,
            num_labels = cfg.data.num_labels,
            num_auxiliary_labels=len(cfg.data.auxiliary_labels),
            dropout=cfg.model.dropout,
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
        params = self.model.parameters()

        decay_parameters = [name for name, parameter in self.model.named_parameters() if ("bias" not in name and 'target_head_auxiliary' not in name)]
        psl_decay_parameters = [name for name, parameter in self.model.named_parameters() if ("bias" not in name and "target_head_auxiliary" in name)]


        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if n in decay_parameters],
                "weight_decay": 5e-3, #self.cfg.training.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if n in psl_decay_parameters],
                "weight_decay": 1e-8, #Small decay on pseudo-likelihood as in Hsu et al.
            },
            {
                "params": [p for n, p in self.model.named_parameters() if (n not in decay_parameters and n not in psl_decay_parameters)],
                "weight_decay": 0.0,
            },
        ]

        optimizer_kwargs = {
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "lr": self.cfg.training.lr
        }

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, **optimizer_kwargs)

        return optimizer



    def training_step(self, batch, batch_idx):
        

        tokens, pad_mask, auxiliary_labels, targets = batch
        
        label_preds, loss = self.model(
            tokens, auxiliary_labels, pad_mask, targets
        )
        self.log("train_label_loss", loss)

        return loss
    

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        

        tokens, pad_mask, auxiliary_labels, targets = batch

        label_preds, loss = self.model(
            tokens, auxiliary_labels, pad_mask, targets
        )
        self.log("val_label_loss", loss)
        return None
    


    def predict_step(self, batch, batch_idx, dataloader_idx=0):

        if self.enable_mc_dropout:
            self.model.train()
        
        tokens, pad_mask, auxiliary_labels, targets = batch

        all_preds = []
        for _ in range(self.mc_dropout_samples):
            label_preds, _ = self.model(
                tokens, auxiliary_labels, pad_mask, targets
            )
            all_preds.append(label_preds)
        
        label_preds = torch.stack(all_preds, dim=-1)
        return label_preds
    

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        
        tokens, pad_mask, auxiliary_labels, targets = batch

        label_preds, loss = self.model(
            tokens, auxiliary_labels, pad_mask, targets
        )
        self.test_spearman.update(label_preds, targets)

        return label_preds
    
    def on_test_epoch_end(self):
        spearman = self.test_spearman.compute()
        self.log("test_spearman", spearman)
        self.test_spearman.reset()


class LightningMLP(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = MLPPropertyPredictor(
            hidden_size=cfg.model.hidden_size,
            embedding_size=cfg.model.embedding_size,
            num_labels = cfg.data.num_labels,
            num_auxiliary_labels=len(cfg.data.auxiliary_labels),
            dropout=cfg.model.dropout,
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
        params = self.model.parameters()
        optimizer = torch.optim.AdamW(params, lr=self.cfg.training.lr)

        return optimizer



    def training_step(self, batch, batch_idx):
        

        tokens, pad_mask, auxiliary_labels, targets = batch
        
        label_preds, loss = self.model(
            tokens, auxiliary_labels, pad_mask, targets
        )
        self.log("train_label_loss", loss)

        return loss
    

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        

        tokens, pad_mask, auxiliary_labels, targets = batch

        label_preds, loss = self.model(
            tokens, auxiliary_labels, pad_mask, targets
        )
        self.log("val_label_loss", loss)
        return None
    


    def predict_step(self, batch, batch_idx, dataloader_idx=0):

        if self.enable_mc_dropout:
            self.model.train()
        
        tokens, pad_mask, auxiliary_labels, targets = batch

        all_preds = []
        for _ in range(self.mc_dropout_samples):
            label_preds, _ = self.model(
                tokens, auxiliary_labels, pad_mask, targets
            )
            all_preds.append(label_preds)
        
        label_preds = torch.stack(all_preds, dim=-1)
        return label_preds
    

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        
        tokens, pad_mask, auxiliary_labels, targets = batch

        label_preds, loss = self.model(
            tokens, auxiliary_labels, pad_mask, targets
        )
        self.test_spearman.update(label_preds, targets)

        return label_preds
    
    def on_test_epoch_end(self):
        spearman = self.test_spearman.compute()
        self.log("test_spearman", spearman)
        self.test_spearman.reset()


