#!/bin/bash
#SBATCH --time=0-01:00:00
#SBATCH --partition=regular
#SBATCH --nodes=1
#SBATCH --mem=10GB
#SBATCH --job-name=visualize_renderer
#SBATCH --output=visualize_renderer.out


module purge
module load Anaconda3/2023.03-1

conda activate pixel-sem-env2

python scripts/visualization/plot_rendered_data.py