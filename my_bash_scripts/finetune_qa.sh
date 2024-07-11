#!/bin/bash
#SBATCH --time=0-08:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:2
#SBATCH --ntasks=10
#SBATCH --mem=100G
#SBATCH --job-name=pixel-experiments-QA-predictions-per-example
#SBATCH --output=pixel-experiments-QA-predictions-per-example.out


module purge
module load Anaconda3/2023.03-1

conda activate pixel-sem-env2

# Optional wandb environment vars
export WANDB_PROJECT="pixel-experiments-QA"

# Settings
export DATASET_NAME="tydiqa"
export DATASET_CONFIG_NAME="secondary_task"
export MODEL="Team-PIXEL/pixel-base" # also works with "bert-base-cased", etc.
export FALLBACK_FONTS_DIR="/scratch/s3919609/data/fallback_fonts"  # let's say this is where we downloaded the fonts to
  
export RUN_NAME="${DATASET_NAME}-$(basename ${MODEL})-${SEQ_LEN}-${BSZ}-${GRAD_ACCUM}-${LR}-${NUM_STEPS}-${SEED}"

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
  --fp16 \
  --half_precision_backend=apex \
  --fallback_fonts_dir=${FALLBACK_FONTS_DIR} \