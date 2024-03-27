#!/bin/bash
#SBATCH --time=0-10:00:00
#SBATCH --partition=regular
#SBATCH --nodes=1
#SBATCH --mem=20GB
#SBATCH --job-name=std_monte_carlo_span_6
#SBATCH --output=std_outputs_span_6.out

# CHANGE LINE: file_handler = logging.FileHandler('std_outputs_mask_0.1.txt')


module purge
module load Anaconda3/2023.09-0

conda activate pixel-sem-env2

python scripts/monte_carlo/monte_carlo_experiments.py \
  --model_name_or_path="Team-PIXEL/pixel-base" \
  --experiment_type="span" \
  --do_std \
  --mask_ratio=0.25 \
  --masking_max_span_length=6 \
  --masking_cumulative_span_weights="0, 0, 0, 0, 0, 1" \
  --span_mask \
  --max_seq_length=256 \