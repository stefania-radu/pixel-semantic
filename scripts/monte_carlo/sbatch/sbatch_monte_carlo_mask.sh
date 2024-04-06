#!/bin/bash
#SBATCH --time=10-00:00:00
#SBATCH --partition=regular
#SBATCH --nodes=1
#SBATCH --mem=20GB
#SBATCH --job-name=std_monte_carlo_mask_0.25
#SBATCH --output=/home2/s3919609/pixel-semantic/scripts/monte_carlo/results/base_experiment_1000/std_outputs_mask_0.25.out

# remember to change the masking spacing   --masking_spacing=0 \

module purge
module load Anaconda3/2023.09-0

conda activate pixel-sem-env2

python scripts/monte_carlo/monte_carlo_experiments.py \
  --input_data_path="scripts/data/uncertainty/test_data_ner_tydiqa_glue_1000.json" \
  --model_name_or_path="Team-PIXEL/pixel-base" \
  --experiment_type="mask_ratio" \
  --ngram_size=1 \
  --do_std \
  --mask_ratio=0.25 \
  --masking_max_span_length=6 \
  --masking_cumulative_span_weights="0.2,0.4,0.6,0.8,0.9,1" \
  --span_mask \
  --max_seq_length=256 \