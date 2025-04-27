#!/bin/bash
#SBATCH --ntasks=1 --cpus-per-task=5 --mem=60G
#SBATCH -p gpu --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH -o /home/zpf738/primo_git/PRIMO/logs/%j_%a.out
#SBATCH -e /home/zpf738/primo_git/PRIMO/logs/%j_%a.err
#SBATCH --job-name=baselines
#SBATCH --array=1-5  # parallelize seeds


# ridge baseline.
python3 scripts/train_baseline.py training.wandb_project=null training.compile=True training.batch_size=4 fine_tuning.few_shot_n=4 hydra.job.name=ridge_4_seed${SLURM_ARRAY_TASK_ID} training.epochs=1500 training.lr=0.01 fine_tuning.seed=$SLURM_ARRAY_TASK_ID
python3 scripts/train_baseline.py training.wandb_project=null training.compile=True training.batch_size=8 fine_tuning.few_shot_n=8 hydra.job.name=ridge_8_seed${SLURM_ARRAY_TASK_ID} training.epochs=1500 training.lr=0.01 fine_tuning.seed=$SLURM_ARRAY_TASK_ID
python3 scripts/train_baseline.py training.wandb_project=null training.compile=True training.batch_size=16 fine_tuning.few_shot_n=16 hydra.job.name=ridge_16_seed${SLURM_ARRAY_TASK_ID} training.epochs=1500 training.lr=0.01 fine_tuning.seed=$SLURM_ARRAY_TASK_ID
python3 scripts/train_baseline.py training.wandb_project=null training.compile=True training.batch_size=32 fine_tuning.few_shot_n=32 hydra.job.name=ridge_32_seed${SLURM_ARRAY_TASK_ID} training.epochs=1500 training.lr=0.01 fine_tuning.seed=$SLURM_ARRAY_TASK_ID
python3 scripts/train_baseline.py training.wandb_project=null training.compile=True training.batch_size=64 fine_tuning.few_shot_n=64 hydra.job.name=ridge_64_seed${SLURM_ARRAY_TASK_ID} training.epochs=1500 training.lr=0.01 fine_tuning.seed=$SLURM_ARRAY_TASK_ID
python3 scripts/train_baseline.py training.wandb_project=null training.compile=True training.batch_size=128 fine_tuning.few_shot_n=128 hydra.job.name=ridge_128_seed${SLURM_ARRAY_TASK_ID} training.epochs=1500 training.lr=0.01 fine_tuning.seed=$SLURM_ARRAY_TASK_ID

# GP baseline.
python3 scripts/train_gp.py training.batch_size=4 fine_tuning.few_shot_n=4 hydra.job.name=kermut_4_seed${SLURM_ARRAY_TASK_ID} fine_tuning.seed=$SLURM_ARRAY_TASK_ID
python3 scripts/train_gp.py training.batch_size=8 fine_tuning.few_shot_n=8 hydra.job.name=kermut_8_seed${SLURM_ARRAY_TASK_ID} fine_tuning.seed=$SLURM_ARRAY_TASK_ID
python3 scripts/train_gp.py training.batch_size=16 fine_tuning.few_shot_n=16 hydra.job.name=kermut_16_seed${SLURM_ARRAY_TASK_ID} fine_tuning.seed=$SLURM_ARRAY_TASK_ID
python3 scripts/train_gp.py training.batch_size=32 fine_tuning.few_shot_n=32 hydra.job.name=kermut_32_seed${SLURM_ARRAY_TASK_ID} fine_tuning.seed=$SLURM_ARRAY_TASK_ID
python3 scripts/train_gp.py training.batch_size=64 fine_tuning.few_shot_n=64 hydra.job.name=kermut_64_seed${SLURM_ARRAY_TASK_ID} fine_tuning.seed=$SLURM_ARRAY_TASK_ID
python3 scripts/train_gp.py training.batch_size=128 fine_tuning.few_shot_n=128 hydra.job.name=kermut_128_seed${SLURM_ARRAY_TASK_ID} fine_tuning.seed=$SLURM_ARRAY_TASK_ID


python3 scripts/train_evolvepro.py training.batch_size=4 fine_tuning.few_shot_n=4 hydra.job.name=evolvepro_4_seed${SLURM_ARRAY_TASK_ID} fine_tuning.seed=${SLURM_ARRAY_TASK_ID}
python3 scripts/train_evolvepro.py training.batch_size=8 fine_tuning.few_shot_n=8 hydra.job.name=evolvepro_8_seed${SLURM_ARRAY_TASK_ID} fine_tuning.seed=${SLURM_ARRAY_TASK_ID}
python3 scripts/train_evolvepro.py training.batch_size=16 fine_tuning.few_shot_n=16 hydra.job.name=evolvepro_16_seed${SLURM_ARRAY_TASK_ID} fine_tuning.seed=${SLURM_ARRAY_TASK_ID}
python3 scripts/train_evolvepro.py training.batch_size=32 fine_tuning.few_shot_n=32 hydra.job.name=evolvepro_32_seed${SLURM_ARRAY_TASK_ID} fine_tuning.seed=${SLURM_ARRAY_TASK_ID}
python3 scripts/train_evolvepro.py training.batch_size=64 fine_tuning.few_shot_n=64 hydra.job.name=evolvepro_64_seed${SLURM_ARRAY_TASK_ID} fine_tuning.seed=${SLURM_ARRAY_TASK_ID}
python3 scripts/train_evolvepro.py training.batch_size=128 fine_tuning.few_shot_n=128 hydra.job.name=evolvepro_128_seed${SLURM_ARRAY_TASK_ID} fine_tuning.seed=${SLURM_ARRAY_TASK_ID}


# MLP
python3 scripts/train_mlp.py training.wandb_project=null model.hidden_size=128 training.compile=False training.batch_size=4 fine_tuning.few_shot_n=4 hydra.job.name=mlp_4_seed${SLURM_ARRAY_TASK_ID} training.epochs=1500 training.lr=0.01 fine_tuning.seed=$SLURM_ARRAY_TASK_ID
python3 scripts/train_mlp.py training.wandb_project=null model.hidden_size=128 training.compile=False training.batch_size=8 fine_tuning.few_shot_n=8 hydra.job.name=mlp_8_seed${SLURM_ARRAY_TASK_ID} training.epochs=1500 training.lr=0.01 fine_tuning.seed=$SLURM_ARRAY_TASK_ID
python3 scripts/train_mlp.py training.wandb_project=null model.hidden_size=128 training.compile=False training.batch_size=16 fine_tuning.few_shot_n=16 hydra.job.name=mlp_16_seed${SLURM_ARRAY_TASK_ID} training.epochs=1500 training.lr=0.01 fine_tuning.seed=$SLURM_ARRAY_TASK_ID
python3 scripts/train_mlp.py training.wandb_project=null model.hidden_size=128 training.compile=False training.batch_size=32 fine_tuning.few_shot_n=32 hydra.job.name=mlp_32_seed${SLURM_ARRAY_TASK_ID} training.epochs=1500 training.lr=0.01 fine_tuning.seed=$SLURM_ARRAY_TASK_ID
python3 scripts/train_mlp.py training.wandb_project=null model.hidden_size=128 training.compile=False training.batch_size=64 fine_tuning.few_shot_n=64 hydra.job.name=mlp_64_seed${SLURM_ARRAY_TASK_ID} training.epochs=1500 training.lr=0.01 fine_tuning.seed=$SLURM_ARRAY_TASK_ID
python3 scripts/train_mlp.py training.wandb_project=null model.hidden_size=128 training.compile=False training.batch_size=128 fine_tuning.few_shot_n=128 hydra.job.name=mlp_128_seed${SLURM_ARRAY_TASK_ID} training.epochs=1500 training.lr=0.01 fine_tuning.seed=$SLURM_ARRAY_TASK_ID