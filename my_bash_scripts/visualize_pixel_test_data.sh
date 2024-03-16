#!/bin/bash
#SBATCH --time=0-10:00:00
#SBATCH --partition=regular
#SBATCH --nodes=1
#SBATCH --mem=20GB
#SBATCH --job-name=experiments_pixel_uncertainty_small
#SBATCH --output=experiments_pixel_uncertainty_small.out


module purge
module load Anaconda3/2023.09-0

conda activate pixel-sem-env2

# Example usage:
python scripts/visualization/visualize_pixel_uncertainty_test_data.py\
  --model_name_or_path="Team-PIXEL/pixel-base" \
  --span_mask \
  --mask_ratio=0.25 \
  --max_seq_length=256 \
  --masking_max_span_length=6 \
  --masking_cumulative_span_weights="0.2,0.4,0.6,0.8,0.9,1" \
  # --rgb=True