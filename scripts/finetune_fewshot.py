import hydra
from omegaconf import DictConfig, OmegaConf
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import lightning as L
import torch
import os
import wandb
import pandas as pd
from scipy.stats import spearmanr

from primo.lightning import LightningSetTransformer
from primo.utils.datamodule import SetDataModule

torch.set_float32_matmul_precision('medium')

RUN_ICL = False
RERUN = False

@hydra.main(version_base=None, config_path="../conf", config_name="config_fewshot")
def run_experiment(cfg : DictConfig) -> None:

    L.seed_everything(123)

    out_dir =  hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    os.makedirs(os.path.join(out_dir, 'predictions'), exist_ok=True)
    print(f"Output directory: {out_dir}")

    # Temporarily disable strict schema
    # we pass this flag to the lightning model so 
    # that it just uses a flat learning rate
    OmegaConf.set_struct(cfg, False)
    # Add the new key
    cfg.is_finetuning = True
    # Re-enable strict schema
    OmegaConf.set_struct(cfg, True)


    dm = SetDataModule(cfg)
    dm.setup() # need to call as we have a non-standard way of accessing the data
    del dm.train_dataset # free up some memory


    # finetune on all test data
    results = {}
    for assay in cfg.data.hold_out.test:

        print(f"Finetuning on {assay}")

        if os.path.exists(os.path.join(out_dir, 'predictions', f"predictions_after_ft_{assay}.csv")):
            print(f"Loading precomputed {assay}")
            
            df = pd.read_csv(os.path.join(out_dir, 'predictions', f"predictions_after_ft_{assay}.csv"))
            perf_after = spearmanr(df['DMS_score'], df['preds'])[0].item()
            indel_rows = df['file'].str.contains('indels')
            df_indel = df.loc[indel_rows]
            df_sub = df.loc[~indel_rows]
            perf_indel_after = spearmanr(df_indel['DMS_score'], df_indel['preds'])[0] if len(df_indel) > 0 else None
            perf_sub_after = spearmanr(df_sub['DMS_score'], df_sub['preds'])[0] if len(df_sub) > 0 else None

            if RUN_ICL:
                df = pd.read_csv(os.path.join(out_dir, 'predictions', f"predictions_before_ft_{assay}.csv"))
                perf_before = spearmanr(df['DMS_score'], df['preds'])[0].item()
                indel_rows = df['file'].str.contains('indels')
                df_indel = df.loc[indel_rows]
                df_sub = df.loc[~indel_rows]
                perf_indel_before = spearmanr(df_indel['DMS_score'], df_indel['preds'])[0] if len(df_indel) > 0 else None
                perf_sub_before = spearmanr(df_sub['DMS_score'], df_sub['preds'])[0] if len(df_sub) > 0 else None
            else:
                perf_before = None
                perf_indel_before = None
                perf_sub_before = None

            
        else:

        # if f"predictions_after_ft_{assay}.csv" in os.listdir(os.path.join(out_dir, 'predictions')) and not RERUN:
        #     print(f"Skipping {assay}")
        #     continue

            train_loader, test_loader = dm.get_finetuning_dataloaders(assay, n=cfg.fine_tuning.few_shot_n, normalization='minmax', seed=cfg.fine_tuning.seed)

            # set up the model
            model = LightningSetTransformer(cfg)

            if cfg.fine_tuning.pretrained_checkpoint_dir is not None:
                model_pt_path = f"{cfg.fine_tuning.pretrained_checkpoint_dir}/last.ckpt"
                print(f"Loading model from {model_pt_path}")
                model = LightningSetTransformer.load_from_checkpoint(model_pt_path, cfg=cfg)
            else:
                print("No pretrained model given, training from scratch. Make sure this is intended.")

            if cfg.training.wandb_project is not None and cfg.fine_tuning.few_shot_n>0:
                wandb_logger = WandbLogger(project=cfg.training.wandb_project, log_model=False, name = cfg.fine_tuning.pretrained_checkpoint_dir + '_' + assay)
                wandb_logger.watch(model)
            else:
                wandb_logger = None
            trainer = Trainer(
                # max_epochs=cfg.training.epochs,
                max_steps=cfg.fine_tuning.num_steps, #* cfg.training.grad_accumulation_steps,
                accelerator='auto',
                devices=cfg.training.devices if torch.cuda.is_available() else 1,
                # check_val_every_n_epoch = 1,
                logger=wandb_logger,
                use_distributed_sampler=False, # TODO work out how to do that properly with iterables
                accumulate_grad_batches=cfg.training.grad_accumulation_steps,
                log_every_n_steps=1,
                # callbacks=[checkpoint_callback],
                num_sanity_val_steps=0, # NOTE this is bugged with many datasets?
                precision=cfg.training.precision,
                gradient_clip_val=cfg.training.clip_grad_norm,
            )

            if wandb_logger is not None:
                # need to do this after the trainer is created, else bugged in distributed mode.
                if trainer.global_rank == 0:
                    wandb_logger.experiment.config.update(OmegaConf.to_container(cfg, resolve=True))

            # get test performance before finetuning
            # perf_before = trainer.test(model, test_loader)
            if RUN_ICL:
                preds = trainer.predict(model, test_loader)
                preds = torch.cat(preds)
                df = test_loader.dataset.assay_dataframe_test.copy()
                df['preds'] = preds.cpu().numpy()
                df.to_csv(os.path.join(out_dir, 'predictions', f"predictions_before_ft_{assay}.csv"))
                perf_before = spearmanr(df['DMS_score'], df['preds'])[0].item()
                # perf_before = [{"test_spearman": 0.0}]

                indel_rows = df['file'].str.contains('indels')
                df_indel = df.loc[indel_rows]
                df_sub = df.loc[~indel_rows]
                perf_indel_before = spearmanr(df_indel['DMS_score'], df_indel['preds'])[0] if len(df_indel) > 0 else None
                perf_sub_before = spearmanr(df_sub['DMS_score'], df_sub['preds'])[0] if len(df_sub) > 0 else None
            else:
                perf_before = None
                perf_indel_before = None
                perf_sub_before = None
        
            if cfg.fine_tuning.few_shot_n >0:
                trainer.fit(model=model, train_dataloaders=train_loader)
                # perf_after = trainer.test(model, test_loader)
                preds = trainer.predict(model, test_loader)
                preds = torch.cat(preds)
                df = test_loader.dataset.assay_dataframe_test.copy()
                df['preds'] = preds.cpu().numpy()
                df.to_csv(os.path.join(out_dir, 'predictions', f"predictions_after_ft_{assay}.csv"))
                perf_after = spearmanr(df['DMS_score'], df['preds'])[0].item()

                indel_rows = df['file'].str.contains('indels')
                df_indel = df.loc[indel_rows]
                df_sub = df.loc[~indel_rows]
                perf_indel_after = spearmanr(df_indel['DMS_score'], df_indel['preds'])[0] if len(df_indel) > 0 else None
                perf_sub_after = spearmanr(df_sub['DMS_score'], df_sub['preds'])[0] if len(df_sub) > 0 else None
            else:
                perf_after = None
                perf_indel_after = None
                perf_sub_after = None

            wandb.finish()

        # results[assay] = {"perf_before": perf_before[0]['test_spearman'], "perf_after": perf_after[0]['test_spearman']}
        results[assay] = {"perf_before": perf_before, "perf_after": perf_after, "perf_indel_before": perf_indel_before, "perf_sub_before": perf_sub_before, "perf_indel_after": perf_indel_after, "perf_sub_after": perf_sub_after}

        # return best_val_score
        pd.DataFrame(results).T.to_csv(os.path.join(out_dir, "finetuning_results.csv"))




if __name__ == "__main__":
    run_experiment()
