#!/bin/bash

# Optional wandb environment vars
export WANDB_PROJECT="pixel-experiments-QA"

# Settings
export DATASET_NAME="tydiqa"
export DATASET_CONFIG_NAME="secondary_task"
export MODEL="Team-PIXEL/pixel-base" # also works with "bert-base-cased", etc.
export FALLBACK_FONTS_DIR="data/fallback_fonts"  # let's say this is where we downloaded the fonts to
export SEQ_LEN=400
export STRIDE=160
export QUESTION_MAX_LEN=128
export BSZ=32
export GRAD_ACCUM=1
export LR=7e-5
export SEED=103
export NUM_STEPS=20000
  
export RUN_NAME="${DATASET_NAME}-$(basename ${MODEL})-${SEQ_LEN}-${BSZ}-${GRAD_ACCUM}-${LR}-${NUM_STEPS}-${SEED}"


TYDIQA_GOLDP_DIR="${HOME}/tydiqa_goldp"
VERSION="1.1"

python scripts/training/run_qa.py \
  --model_name_or_path=${MODEL} \
  --dataset_name=${DATASET_NAME} \
  --dataset_config_name=${DATASET_CONFIG_NAME} \
  --do_predict \
  --remove_unused_columns=False \
  --test_file="/scratch/s3919609/Pixel-stuff/tydiqa_goldp/tydiqa-goldp-v1.1-dev.json" \
  --dropout_prob=0.15 \
  --max_seq_length=${SEQ_LEN} \
  --question_max_length=${QUESTION_MAX_LEN} \
  --doc_stride=${STRIDE} \
  --max_steps=${NUM_STEPS} \
  --num_train_epochs=10 \
  --early_stopping \
  --early_stopping_patience=5 \
  --per_device_train_batch_size=${BSZ} \
  --gradient_accumulation_steps=${GRAD_ACCUM} \
  --learning_rate=${LR} \
  --warmup_steps=100 \
  --run_name=${RUN_NAME} \
  --output_dir=${RUN_NAME} \
  --overwrite_output_dir \
  --overwrite_cache \
  --logging_strategy=steps \
  --logging_steps=100 \
  --evaluation_strategy=steps \
  --eval_steps=500 \
  --save_strategy=steps \
  --save_steps=500 \
  --save_total_limit=5 \
  --report_to=wandb \
  --log_predictions \
  --load_best_model_at_end=True \
  --metric_for_best_model="eval_f1" \
  --fp16 \
  --half_precision_backend=apex \
  --fallback_fonts_dir=${FALLBACK_FONTS_DIR} \
  --seed=${SEED}
