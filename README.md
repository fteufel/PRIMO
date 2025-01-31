# PRIMO

A transformer model for performing set-based prediction of protein fitness using in-context learning and test-time training.

## Getting started

1. Install
```bash
pip install -e .
```


## Data prep

These scripts rely on hardcoded paths pointing to /data. They require materials that can be downloaded from ProteinGym.
```bash

for file in data/unformatted/*.csv
do
  python3 scripts/compute_fitness_progen.py "${file}" "${file}" --all_columns --Progen2_model_name_or_path misc_checkpoints/progen2-medium
done

python3 scripts/reformat_data.py
python3 scripts/add_random_cv_folds.py

# this here fills in progen scores for stuff that doesn't have it precomputed, and writes to a single file.
python3 scripts/compute_fitness_progen.py data/all_samples.csv data/progen_scores.csv --Progen2_model_name_or_path misc_checkpoints/progen2-medium

```

## Training

```bash
python3 scripts/train.py
```

## Evaluation

```bash

# PRIMO
pretrained_checkpoint_dir=checkpoints/PRIMO # trained model from previous step
n=8 # few-shot size
n_test=$((n+1))
python3 scripts/finetune_fewshot.py --config-name=config_fewshot\
 model.attn_method=pooled data.sets_per_epoch=1000\
 fine_tuning.pretrained_checkpoint_dir=$pretrained_checkpoint_dir\
 fine_tuning.few_shot_n=$n\
 data.set_size=$n\
 data.window_size=null\
 data.set_size_test=$n_test\
 training.batch_size=1\
 training.grad_accumulation_steps=12


# ridge baseline.
python3 scripts/train_baseline.py training.batch_size=$n fine_tuning.few_shot_n=$n training.epochs=1500 training.lr=0.01

# GP baseline.
python3 scripts/train_gp.py training.batch_size=$n fine_tuning.few_shot_n=$n
```