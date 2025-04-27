#!/bin/bash
#SBATCH --ntasks=1 --cpus-per-task=5 --mem=60G
#SBATCH -p gpu --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH -o /home/zpf738/primo_git/PRIMO/logs/%j_%a.out
#SBATCH -e /home/zpf738/primo_git/PRIMO/logs/%j_%a.err
#SBATCH --job-name=primo
#SBATCH --array=0-35  # Adjust based on the number of tests you want to run

# Print GPU information
nvidia-smi

# Source environment and change directory
source /home/zpf738/.bashrc
conda activate pgf
cd /home/zpf738/primo_git/PRIMO

# List of test entries from the YAML
TEST_ENTRIES=(
    "AMFR_HUMAN_Tsuboyama_2023_4G3O"
    "RCD1_ARATH_Tsuboyama_2023_5OAO"
    "SR43C_ARATH_Tsuboyama_2023_2N88"
    "FECA_ECOLI_Tsuboyama_2023_2D1U"
    "PKN1_HUMAN_Tsuboyama_2023_1URF"
    "CSN4_MOUSE_Tsuboyama_2023_1UFM"
    "SPA_STAAU_Tsuboyama_2023_1LP1"
    "NKX31_HUMAN_Tsuboyama_2023_2L9R"
    "EPHB2_HUMAN_Tsuboyama_2023_1F0M"
    "SQSTM_MOUSE_Tsuboyama_2023_2RRU"
    "MAFG_MOUSE_Tsuboyama_2023_1K1V"
    "SCIN_STAAR_Tsuboyama_2023_2QFF"
    "DNJA1_HUMAN_Tsuboyama_2023_2LO1"
    "VRPI_BPT7_Tsuboyama_2023_2WNM"
    "ESTA_BACSU_Nutschel_2020"
    "BLAT_ECOLX_Deng_2012"
    "BLAT_ECOLX_Jacquier_2013"
    "BLAT_ECOLX_Stiffler_2015"
    "BLAT_ECOLX_Firnberg_2014"
    "VKOR1_HUMAN_Chiasson_2020_activity"
    "VKOR1_HUMAN_Chiasson_2020_abundance"
    "Q8WTC7_9CNID_Somermeyer_2022"
    "CASP3_HUMAN_Roychowdhury_2020"
    "D7PM05_CLYGR_Somermeyer_2022"
    "GFP_AEQVI_Sarkisyan_2016"
    "BLAT_ECOLX_Gonzalez_2019"
    "DLG4_HUMAN_Faure_2021"
    "RL40A_YEAST_Mavor_2016"
    "RL40A_YEAST_Roscoe_2014"
    "DYR_ECOLI_Nguyen_2023"
    "DLG4_RAT_McLaughlin_2012"
    "RL40A_YEAST_Roscoe_2013"
    "GRB2_HUMAN_Faure_2021"
    "DYR_ECOLI_Thompson_2019"
    "CAPSD_AAV2S_Sinai_2021"
)

# Get the current test entry based on the array task ID
CURRENT_TEST=${TEST_ENTRIES[$SLURM_ARRAY_TASK_ID]}

# Default parameters
n_steps=25
pretrained_checkpoint_dir=outputs/default_primo

# Iterate over n values
for n in 4 8 16 23 64 128
do
    n_test=$((n+1))
    set_size=$((n<32?n:32))

    # Iterate over seeds 1 to 5
    for seed in 1 2 3 4 5
    do
        # Run the Python script with the current test entry
        python3 scripts/finetune_fewshot.py --config-name=config_fewshot \
         model.attn_method=pooled data.sets_per_epoch=1000 \
         fine_tuning.pretrained_checkpoint_dir=$pretrained_checkpoint_dir \
         fine_tuning.num_steps=$n_steps \
         hydra.job.name=primo_${n}shot_${n_steps}steps_seed${seed} \
         fine_tuning.few_shot_n=$n \
         data.set_size=$set_size \
         training.wandb_project=null \
         data.window_size=null \
         data.set_size_test=$n_test \
         fine_tuning.seed=$seed \
         training.batch_size=2\
         training.grad_accumulation_steps=6\
         hold_out.test=["$CURRENT_TEST"] \
         training.compile=False
        
    done
done