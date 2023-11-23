#!/bin/bash
#SBATCH --time=0-00:10:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=5
#SBATCH --ntasks=10
#SBATCH --job-name=pixel_test_gpu
#SBATCH --output=pixel.out


module purge
module load Anaconda3/2023.03-1

conda activate pixel-sem-env

bash $HOME/pixel-semantic/my_bash_scripts/finetune_sc.sh