#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=160GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=/home2/s3919609/pixel-semantic/experiments/pretraining/last_model/%j_0_log.out
#SBATCH --partition=gpu
#SBATCH --time=3-00:00:00
#SBATCH --job-name=pretraining_multilingual_3_day_1_gpu


module purge
module load Anaconda3/2023.03-1
conda activate pixel-sem-env2

export WANDB_PROJECT='pixel-multilingual-pretraining'

python test_pretraining.py --job_dir=/home2/s3919609/pixel-semantic/experiments/pretraining/last_model/model  --prototype_config_name=scratch_noto_span0.25-dropout_ngram2   --training_config_name=fp16_apex_bs32_multilingual
