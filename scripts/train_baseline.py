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

from primo.lightning_baseline import LightningAugmentedRidgeRegression
from primo.utils.datamodule import SetDataModule

torch.set_float32_matmul_precision('medium')

@hydra.main(version_base=None, config_path="../conf", config_name="config_fewshot")
def run_experiment(cfg : DictConfig) -> None:

    L.seed_everything(123)

    out_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    os.makedirs(os.path.join(out_dir, "predictions"), exist_ok=True)
    print(f'Saving outputs in {out_dir}')

    # Temporarily disable strict schema
    # we pass this flag to the lightning model so 
    # that it just uses a flat learning rate
    OmegaConf.set_struct(cfg, False)
    # Add the new key
    cfg.is_finetuning = True
    cfg.training.dropout = 0.0 # no dropout in Hsu et al.
    # Re-enable strict schema
    OmegaConf.set_struct(cfg, True)


    dm = SetDataModule(cfg)
    dm.setup() # need to call as we have a non-standard way of accessing the data

    # finetune on all test data
    results = {}
    parameters = {}
    for assay in cfg.data.hold_out.test:

        # if assay not in ['EPHB2_HUMAN_Tsuboyama_2023_1F0M']:
        #     continue

        print(f"Training on {assay}")

        train_loader, test_loader = dm.get_baseline_dataloaders(assay, n=cfg.fine_tuning.few_shot_n, data_selection_mode='random_increasing', seed=cfg.fine_tuning.seed)

        # set up the model
        model = LightningAugmentedRidgeRegression(cfg)

        if cfg.training.wandb_project is not None:
            wandb_logger = WandbLogger(project=cfg.training.wandb_project, log_model=False, name =  'baseline_' + assay)
            wandb_logger.watch(model)
        trainer = Trainer(
            max_epochs=cfg.training.epochs,
            # max_steps=cfg.fine_tuning.num_steps * cfg.training.grad_accumulation_steps,
            accelerator='gpu',
            # devices=cfg.training.devices if torch.cuda.is_available() else 1,
            logger=wandb_logger if cfg.training.wandb_project is not None else None,
            use_distributed_sampler=False, # TODO work out how to do that properly with iterables
            # accumulate_grad_batches=cfg.training.grad_accumulation_steps,
            log_every_n_steps=10,
            num_sanity_val_steps=0, # NOTE this is bugged with many datasets?
            precision=cfg.training.precision,
            gradient_clip_val=cfg.training.clip_grad_norm,
        )

        if cfg.training.wandb_project is not None:
            # need to do this after the trainer is created, else bugged in distributed mode.
            if trainer.global_rank == 0:
                wandb_logger.experiment.config.update(OmegaConf.to_container(cfg, resolve=True))


        trainer.fit(model=model, train_dataloaders=train_loader)
        # perf_after = trainer.test(model, test_loader)
        preds = trainer.predict(model, test_loader)
        preds = torch.cat(preds, dim=0).cpu().numpy()
        df = test_loader.dataset.assay_dataframe
        df['pred'] = preds
        perf = spearmanr(df['DMS_score'], df['pred'])[0]

        indel_rows = df['file'].str.contains('indels')
        df_indel = df.loc[indel_rows]
        df_sub = df.loc[~indel_rows]
        perf_indel = spearmanr(df_indel['DMS_score'], df_indel['pred'])[0] if len(df_indel) > 0 else None
        perf_sub = spearmanr(df_sub['DMS_score'], df_sub['pred'])[0] if len(df_sub) > 0 else None
        wandb.finish()

        params_bias = model.model.target_head.bias.cpu().detach().numpy() # 1,1
        params_emb = model.model.target_head.weight.cpu().detach().numpy() # 1, 1280
        params_aux = model.model.target_head_auxiliary.weight.cpu().detach().numpy() # 1, 1
        parameters[assay] = {'bias': params_bias.tolist()[0]}
        parameters[assay].update({f'emb_{i}': v for i, v in enumerate(params_emb[0].tolist())})
        parameters[assay].update({f'aux_{i}': v for i, v in enumerate(params_aux[0].tolist())})

        # results[assay] = {"perf_after": perf_after[0]['test_spearman']}
        results[assay] = {"perf_after": perf, 'perf_indel': perf_indel, 'perf_sub': perf_sub}

        # save predictions
        df.to_csv(os.path.join(out_dir, 'predictions', f"predictions_{assay}.csv"))

        pd.DataFrame(results).T.to_csv(os.path.join(out_dir, "test_results.csv"))

        pd.DataFrame(parameters).T.to_csv(os.path.join(out_dir, "parameters.csv"))

        # model teardown to free up gpu memory
        # because we use hacky non-registration of the esm weights internally 
        # (to avoid saving them with the checkpoint), we need to clean it up
        # explicitly here
        del model.model.cached_esm.cache
        del model.model.esm_as_list[0]

        del model




if __name__ == "__main__":
    run_experiment()
