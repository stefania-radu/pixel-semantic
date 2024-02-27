#!/bin/bash
#SBATCH --time=0-08:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:2
#SBATCH --ntasks=10
#SBATCH --mem=20G
#SBATCH --job-name=pixel-experiments-QA-ensemble-tydiqa-104
#SBATCH --output=pixel-experiments-QA-ensemble-tydiqa-104.out


module purge
module load Anaconda3/2023.09-0

conda activate pixel-sem-env2

nvcc --version
nvidia-smi

bash $HOME/pixel-semantic/my_bash_scripts/finetune_qa.sh