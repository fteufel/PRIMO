import hydra
from omegaconf import DictConfig, OmegaConf
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import lightning as L
import torch

from pairtransformer.lightning import LightningPairTransformer
from pairtransformer.utils.datamodule import SetDataModule

torch.set_float32_matmul_precision('medium')

print('Attention backend:')
print(torch.backends.cuda.flash_sdp_enabled())
# True
print(torch.backends.cuda.mem_efficient_sdp_enabled())
# True
print(torch.backends.cuda.math_sdp_enabled())
# True

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def run_experiment(cfg : DictConfig) -> None:
    # print(OmegaConf.to_yaml(cfg))

    L.seed_everything(123)


    # set up the data
    dm = SetDataModule(cfg)

    # set up the model
    print("Loading model")
    model = LightningPairTransformer(cfg)

    checkpoint_callback = ModelCheckpoint(
        monitor=cfg.training.stopping_criterion,  # Specify the metric to monitor
        mode='min' if 'mse' in cfg.training.stopping_criterion else 'max',  # Specify the direction of improvement
        save_top_k=1,  # Save the best checkpoint only,
        save_last=True,
        dirpath = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    )
    print(f'Saving checkpoints at {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}')


    if cfg.training.wandb_project is not None:
        wandb_logger = WandbLogger(project=cfg.training.wandb_project, log_model=False)
        wandb_logger.watch(model)
    trainer = Trainer(
        max_epochs=cfg.training.epochs,
        accelerator='auto',
        devices=cfg.training.devices if torch.cuda.is_available() else 1,
        check_val_every_n_epoch = 1,
        logger=wandb_logger if cfg.training.wandb_project is not None else None,
        use_distributed_sampler=False, # TODO work out how to do that properly with iterables
        accumulate_grad_batches=cfg.training.grad_accumulation_steps,
        log_every_n_steps=5,
        callbacks=[checkpoint_callback],
        num_sanity_val_steps=0, # NOTE this is bugged with many datasets?
        precision=cfg.training.precision,
        gradient_clip_val=cfg.training.clip_grad_norm,
    )

    if cfg.training.wandb_project is not None:
        # need to do this after the trainer is created, else bugged in distributed mode.
        if trainer.global_rank == 0:
            wandb_logger.experiment.config.update(OmegaConf.to_container(cfg, resolve=True))

    # print('debug validation')
    # import ipdb; ipdb.set_trace() # 113 datasets
    # trainer.validate(model, datamodule=dm, verbose=False)
    # import ipdb; ipdb.set_trace()

    trainer.fit(model, dm)

    if cfg.training.long_seq_epochs > 0:
        # now we train on the long sequences
        factor = cfg.data.window_size_longer // cfg.data.window_size
        # scaling is quadratic - need to reduce the batch size by the square
        new_batch_size = cfg.training.batch_size // (factor ** 2)
        new_grad_accumulation_steps = cfg.training.grad_accumulation_steps * (factor ** 2)

        # override the datamodule config
        dm.cfg.training.batch_size = new_batch_size
        dm.train_dataset.min_pad_length = cfg.data.window_size_longer
        dm.val_pair_dataset.min_pad_length = cfg.data.window_size_longer
        from pairtransformer.utils.set_samplers import CombinedProbabilitySampler
        if type(dm.train_dataset.pair_sampler) == CombinedProbabilitySampler:
            for sampler in dm.train_dataset.pair_sampler.samplers:
                sampler.window_size = cfg.data.window_size_longer
        else:
            dm.train_dataset.pair_sampler.window_size = cfg.data.window_size_longer

        if type(dm.val_pair_dataset.pair_sampler) == CombinedProbabilitySampler:
            for sampler in dm.val_pair_dataset.pair_sampler.samplers:
                sampler.window_size = cfg.data.window_size_longer
        else:
            dm.val_pair_dataset.pair_sampler.window_size = cfg.data.window_size_longer

        trainer = Trainer(
            max_epochs=cfg.training.long_seq_epochs,
            accelerator='auto',
            devices=cfg.training.devices if torch.cuda.is_available() else 1,
            check_val_every_n_epoch = 1,
            logger=wandb_logger if cfg.training.wandb_project is not None else None,
            use_distributed_sampler=False, # TODO work out how to do that properly with iterables
            accumulate_grad_batches=new_grad_accumulation_steps,
            log_every_n_steps=5,
            callbacks=[checkpoint_callback],
            num_sanity_val_steps=0, # NOTE this is bugged with many datasets?
            precision=cfg.training.precision,
            gradient_clip_val=cfg.training.clip_grad_norm,
        )

        trainer.fit(model, dm)


    best_val_score = checkpoint_callback.best_model_score
    best_model_path = checkpoint_callback.best_model_path
    best_model = model.__class__.load_from_checkpoint(best_model_path, cfg=cfg)

    # trainer.test(best_model, datamodule=dm)

    # dump preds in 3-line format
    # save_predictions(dm, trainer, best_model, hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)


    # for hyperparameter optimization
    return best_val_score



if __name__ == "__main__":
    run_experiment()
