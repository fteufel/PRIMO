import hydra
from omegaconf import DictConfig
import lightning as L
import torch
import os
import pandas as pd
from scipy.stats import spearmanr
from esm.pretrained import load_model_and_alphabet
from tqdm.auto import tqdm
from sklearn.ensemble import RandomForestRegressor
import numpy as np

from primo.utils.datamodule import SetDataModule
from primo.modules.baselines import CachedESMModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_float32_matmul_precision('medium')


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
    esm_embedder = CachedESMModel()

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
        # train_pad_masks = torch.cat(all_pad_masks, dim=0).to(device)
        # x_zeroshot = torch.cat(all_aux_labels, dim=0).to(device)
        y = torch.cat(all_label_targets, dim=0).to(device)
        x_embeddings = esm_embedder(esm, train_inputs).mean(dim=1)


        # set up the model
        # https://github.com/idmjky/EvolvePro/blob/main/top_layer.py
        # If “auto”, then max_features=n_features.
        model = RandomForestRegressor(n_estimators=100, criterion='friedman_mse', max_depth=None, min_samples_split=2,
                                      min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None,#'auto',
                                      max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False,
                                      n_jobs=None, random_state=1, verbose=0, warm_start=False, ccp_alpha=0.0,
                                      max_samples=None)
        
        model.fit(x_embeddings.cpu().numpy(), y.cpu().numpy())
     

     
        preds = []
        with torch.no_grad():
            for batch in tqdm(test_loader):
                seq_tokens, pad_mask, auxiliary_labels, label_target = batch
                x_test = esm_embedder(esm, seq_tokens.to(device)).mean(dim=1)
                y_preds = model.predict(x_test.cpu().numpy())
                preds.append(y_preds)

        # preds = torch.cat(preds, dim=0).numpy()
        preds = np.concatenate(preds)

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






if __name__ == "__main__":
    run_experiment()
