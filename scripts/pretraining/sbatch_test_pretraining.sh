#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=a100:4
#SBATCH --mem=160GB
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --output=/home2/s3919609/pixel-semantic/experiments/pretraining/%j/%j_0_log.out
#SBATCH --partition=gpu
#SBATCH --time=0-00:10:00
#SBATCH --job-name=pretraining_multilingual_big


module purge
module load Anaconda3/2023.03-1
conda activate pixel-sem-env2

export WANDB_PROJECT='pixel-multilingual-pretraining'

python test_pretraining.py --job_dir=/home2/s3919609/pixel-semantic/experiments/pretraining/%j/model   --prototype_config_name=scratch_noto_span0.25-dropout   --training_config_name=fp16_apex_bs32_multilingual_test