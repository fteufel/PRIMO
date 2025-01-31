import hydra
from omegaconf import DictConfig, OmegaConf
import lightning as L
import torch
import os
import wandb
import pandas as pd
from scipy.stats import spearmanr
from esm.pretrained import load_model_and_alphabet
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.priors import HalfCauchyPrior
import gpytorch
from tqdm.auto import tqdm

from primo.modules.almost_kermut import KermutGP, optimize_gp
from primo.utils.datamodule import SetDataModule
from primo.modules.baselines import CachedESMModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_float32_matmul_precision('medium')


    # gp.eval()
    # likelihood.eval()

    # x_test = tuple([x for x in test_inputs if x is not None])
    # y_test = test_targets.detach().cpu().numpy()

    # with torch.no_grad():
    #     # Predictive distribution
    #     y_preds_dist = likelihood(gp(*x_test))
    #     y_preds_mean = y_preds_dist.mean.detach().cpu().numpy()
    #     y_preds_var = y_preds_dist.covariance_matrix.diag().detach().cpu().numpy()

    #     df_out.loc[test_idx, "fold"] = test_fold
    #     df_out.loc[test_idx, "y"] = y_test
    #     df_out.loc[test_idx, "y_pred"] = y_preds_mean
    #     df_out.loc[test_idx, "y_var"] = y_preds_var

    # return df_out


@hydra.main(version_base=None, config_path="../conf", config_name="config_fewshot")
def run_experiment(cfg : DictConfig) -> None:

    L.seed_everything(123)

    out_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    os.makedirs(os.path.join(out_dir, "predictions"), exist_ok=True)
    print(f'Saving outputs in {out_dir}')


    dm = SetDataModule(cfg)
    dm.setup() # need to call as we have a non-standard way of accessing the data

    esm, _ = load_model_and_alphabet('esm2_t33_650M_UR50D')
    esm.to(device)
    esm_embedder =  CachedESMModel()

    # train on all test data
    results = {}
    parameters = {}
    for assay in cfg.data.hold_out.test:


        print(f"Training on {assay}")

        train_loader, test_loader = dm.get_baseline_dataloaders(assay, n=cfg.fine_tuning.few_shot_n, data_selection_mode='random_increasing', normalization='z_score', seed=cfg.fine_tuning.seed)

        # put together data
        all_seq_tokens = []
        all_pad_masks = []
        all_aux_labels = []
        all_label_targets = []

        # Iterate through the DataLoader and collect the data and labels
        for batch in train_loader:
            seq_tokens, pad_mask, auxiliary_labels, label_target = batch
            all_seq_tokens.append(seq_tokens)
            all_pad_masks.append(pad_mask)
            all_aux_labels.append(auxiliary_labels)
            all_label_targets.append(label_target)


        # Concatenate all the data and labels into single tensors
        train_inputs = torch.cat(all_seq_tokens, dim=0).to(device)
        train_pad_masks = torch.cat(all_pad_masks, dim=0).to(device)
        x_zeroshot = torch.cat(all_aux_labels, dim=0).to(device)
        y = torch.cat(all_label_targets, dim=0).to(device)
        x_embeddings = esm_embedder(esm, train_inputs).mean(dim=1)


        # set up the model
        noise_prior = HalfCauchyPrior(scale=0.1)
        likelihood = GaussianLikelihood(noise_prior=noise_prior).to(device)
        model = KermutGP((x_embeddings, x_zeroshot), y, likelihood).to(device)


        optimize_gp(model, likelihood, (x_embeddings, x_zeroshot), y)


        model.eval()
        likelihood.eval()
        preds = []


        with torch.no_grad(), gpytorch.settings.skip_posterior_variances():
            for batch in tqdm(test_loader):
                seq_tokens, pad_mask, auxiliary_labels, label_target = batch
                x_test = esm_embedder(esm, seq_tokens.to(device)).mean(dim=1)
                f_preds_dist = model(x_test, auxiliary_labels.to(device))
                y_preds = likelihood(f_preds_dist).mean.detach().cpu()
                preds.append(y_preds)

        preds = torch.cat(preds, dim=0).numpy()

        df = test_loader.dataset.assay_dataframe
        df['pred'] = preds
        perf = spearmanr(df['DMS_score'], df['pred'])[0]

        indel_rows = df['file'].str.contains('indels')
        df_indel = df.loc[indel_rows]
        df_sub = df.loc[~indel_rows]
        perf_indel = spearmanr(df_indel['DMS_score'], df_indel['pred'])[0] if len(df_indel) > 0 else None
        perf_sub = spearmanr(df_sub['DMS_score'], df_sub['pred'])[0] if len(df_sub) > 0 else None

        results[assay] = {"perf_after": perf, 'perf_indel': perf_indel, 'perf_sub': perf_sub}

        # save predictions
        df.to_csv(os.path.join(out_dir, 'predictions', f"predictions_{assay}.csv"))

        pd.DataFrame(results).T.to_csv(os.path.join(out_dir, "test_results.csv"))


        # # model teardown to free up gpu memory
        # # because we use hacky non-registration of the esm weights internally 
        # # (to avoid saving them with the checkpoint), we need to clean it up
        # # explicitly here
        # del model.model.cached_esm.cache
        # del model.model.esm_as_list[0]

        # del model




if __name__ == "__main__":
    run_experiment()
