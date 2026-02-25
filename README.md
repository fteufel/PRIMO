# PRIMO

A transformer model for performing set-based prediction of protein fitness using in-context learning and test-time training.

## Getting started

1. Install
```bash
pip install -e .
```


## Data prep

These scripts rely on hardcoded paths pointing to `/data`. They require materials that can be [downloaded from ProteinGym](https://proteingym.org/download). Please download all ProteinGym DMS (csv formatted, `DMS_ProteinGym_substitutions.zip` and `DMS_ProteinGym_indels.zip`) and place them in `data/unformatted`. 

The random split needs to be placed in `data/cv_folds`. Download 
https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/cv_folds_singles_substitutions.zip,
https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/cv_folds_multiples_substitutions.zip,
https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/cv_folds_indels.zip

Next we precompute ProGen 0-shot scores for each sample. Download ProGen2-medium (https://github.com/enijkamp/progen2) and place it at e.g. `misc_checkpoints`. 
```bash

for file in data/unformatted/*.csv
do
  python3 scripts/compute_fitness_progen.py "${file}" "${file}" --all_columns --Progen2_model_name_or_path misc_checkpoints/progen2-medium
done
python3 scripts/compute_fitness_progen.py data/all_wt.csv data/progen_scores_wt.csv --Progen2_model_name_or_path misc_checkpoints/progen2-medium --all_columns
# rename `score_progen2-medium` in row 0 to label:
sed -i '1s/score_progen2-medium/label/' data/progen_scores_wt.csv

# reformat data for loading
python3 scripts/reformat_data.py
python3 scripts/add_random_cv_folds.py

# this here fills in progen scores for stuff that doesn't have it precomputed, and writes to a single file that we load for 0shot scores.
python3 scripts/compute_fitness_progen.py data/all_samples.csv data/progen_scores.csv --Progen2_model_name_or_path misc_checkpoints/progen2-medium

```

## Training

This command trains the PRIMO model, using a batch size of 4*3=12. Depending on GPU memory available, the factoring can be adjusted to improve speed.
```bash
python3 scripts/train.py training.grad_accumulation_steps=6 training.batch_size=2 hydra.job.name=default_primo training.compile=False
```

### Ablation models
```bash
# PRIMO with the Metalic split
python3 scripts/train.py --config_name=config_fewshot_dirty training.grad_accumulation_steps=4 training.batch_size=3 hydra.job.name=badsplit_primo training.compile=False
# PRIMO with MSE loss instead of preference loss
python3 scripts/train.py training.grad_accumulation_steps=6 training.batch_size=2 hydra.job.name=mse_primo training.compile=True model.loss_fn=mse

python3 scripts/train.py training.grad_accumulation_steps=6 training.batch_size=2 hydra.job.name=no0shot_primo training.compile=False data.ablate_auxiliary=True
python3 scripts/train.py training.grad_accumulation_steps=6 training.batch_size=2 hydra.job.name=notype_primo training.compile=False data.ablate_type=True

```


## Evaluation

Slurm scripts are provided for evaluating the model on the test set. The model checkpoint should be specified in the slurm script.


```bash
sbatch submit_baselines.sh # 5-array job that runs all baselines, parallelizing over seeds
sbatch submit_primo_array.sh # 36-array job that runs primo, parallelizing over datasets
sbatch submit_mseprimo_array.sh # 36-array job that runs primo, parallelizing over datasets

```

