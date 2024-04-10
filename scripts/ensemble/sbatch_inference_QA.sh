#!/bin/bash
#SBATCH --time=0-08:00:00
#SBATCH --partition=regular
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --mem=20G
#SBATCH --job-name=QA-from-paper
#SBATCH --output=QA-from-paper.out


module purge
module load Anaconda3/2023.09-0

conda activate pixel-sem-env2

export DATASET_NAME="tydiqa"
export DATASET_CONFIG_NAME="secondary_task"
export MODEL="Team-PIXEL/pixel-base-finetuned-squadv1"
export FALLBACK_FONTS_DIR="data/fallback_fonts"  # let's say this is where we downloaded the fonts to

python scripts/training/run_qa.py \
  --model_name_or_path=${MODEL} \
  --dataset_name=${DATASET_NAME} \
  --dataset_config_name=${DATASET_CONFIG_NAME} \
  --remove_unused_columns=False \
  --do_eval \
  --per_device_eval_batch_size=128 \
  --max_seq_length=400 \
  --doc_stride=160 \
  --output_dir=test-qa \
  --report_to=wandb \
  --overwrite_cache \
  --metric_for_best_model=eval_f1 \
  --fallback_fonts_dir=${FALLBACK_FONTS_DIR}  # not necessary here, but good to check that it works
