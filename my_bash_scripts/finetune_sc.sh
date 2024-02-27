#!/bin/bash

# Note on GLUE: 
# We found that for some of the tasks (e.g. MNLI), PIXEL can get stuck in a bad local optimum
# A clear indicator of this is when the training loss is not decreasing substantially within the first 1k-3k steps
# If this happens, you can tweak the learning rate slightly, increase the batch size,
# change rendering backends, or often even just the random seed
# We are still trying to find the optimal training recipe for PIXEL on these tasks,
# the recipes used in the paper may not be the best ones out there

# Optional wandb environment vars
export WANDB_PROJECT="pixel-experiments-SC"

# change the cache so that we dont run out of space
export HF_HOME="/scratch/.cache/huggingface"
export HF_DATASETS_CACHE="/scratch/.cache/huggingface/datasets"
export TRANSFORMERS_CACHE="/scratch/.cache/huggingface/models"

# Settings
export TASK="cola"
export MODEL="Team-PIXEL/pixel-base" # also works with "bert-base-cased", "roberta-base", etc.
export RENDERING_BACKEND="pygame"  # Consider trying out both "pygame" and "pangocairo" to see which one works best
export POOLING_MODE="pma8" # Can be "mean", "max", "cls", or "pma1" to "pma8"
export SEQ_LEN=256
export BSZ=64 # it was 64 but it does not run on my machine
export GRAD_ACCUM=8  # We found that higher batch sizes can sometimes make training more stable
export LR=5e-5
export SEED=104
export NUM_STEPS=10000
  
export RUN_NAME="${TASK}-$(basename ${MODEL})-${POOLING_MODE}-${RENDERING_BACKEND}-${SEQ_LEN}-${BSZ}-${GRAD_ACCUM}-${LR}-${NUM_STEPS}-${SEED}"

python scripts/training/run_glue.py \
  --model_name_or_path=${MODEL} \
  --task_name=${TASK} \
  --rendering_backend=${RENDERING_BACKEND} \
  --pooling_mode=${POOLING_MODE} \
  --pooler_add_layer_norm=True \
  --remove_unused_columns=False \
  --do_train \
  --do_eval \
  --do_predict \
  --dropout_prob=0.1 \
  --max_seq_length=${SEQ_LEN} \
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
  --metric_for_best_model="eval_matthews_correlation" \
  --half_precision_backend=apex \
  --seed=${SEED} \
  --fp16
   
    