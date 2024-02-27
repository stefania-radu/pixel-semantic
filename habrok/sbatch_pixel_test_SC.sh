#!/bin/bash
#SBATCH --time=0-02:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:2
#SBATCH --ntasks=10
#SBATCH --mem=30G
#SBATCH --job-name=pixel-experiments-SC-ensemble-cola-104
#SBATCH --output=pixel-experiments-SC-ensemble-cola-104.out


module purge
module load Anaconda3/2023.03-1

conda activate pixel-sem-env2

nvcc --version
nvidia-smi

bash $HOME/pixel-semantic/my_bash_scripts/finetune_sc.sh