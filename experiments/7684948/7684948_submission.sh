#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=2
#SBATCH --job-name=pixel
#SBATCH --mem=80GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --output=/home2/s3919609/pixel-semantic/experiments/%j/%j_0_log.out
#SBATCH --partition=gpu
#SBATCH --signal=USR2@120
#SBATCH --time=4320
#SBATCH --wckey=submitit

# setup
#SBATCH --time=0-00:10:00
#SBATCH --job-name=pretraining_multilingual_short
#SBATCH --output=pretraining_multilingual_short.out
module purge
module load Anaconda3/2023.03-1
conda activate pixel-sem-env2
export WANDB_PROJECT='pixel-multilingual-pretraining'

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /home2/s3919609/pixel-semantic/experiments/%j/%j_%t_log.out /scratch/s3919609/conda/envs/pixel-sem-env2/bin/python -u -m submitit.core._submit /home2/s3919609/pixel-semantic/experiments/%j
