#!/bin/bash
#SBATCH --time=0-08:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:2
#SBATCH --ntasks=10
#SBATCH --mem=20G
#SBATCH --job-name=pixel-experiments-QA-ensemble-tydiqa-100
#SBATCH --output=pixel-experiments-QA-ensemble-tydiqa-100.out


module purge
module load Anaconda3/2023.03-1

conda activate pixel-sem-env2

nvcc --version
nvidia-smi

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
export BSZ=16
export GRAD_ACCUM=1
export LR=7e-5
export SEED=100
export NUM_STEPS=20000
  
export RUN_NAME="${DATASET_NAME}-$(basename ${MODEL})-${SEQ_LEN}-${BSZ}-${GRAD_ACCUM}-${LR}-${NUM_STEPS}-${SEED}"

python scripts/training/run_qa.py \
  --model_name_or_path=${MODEL} \
  --dataset_name=${DATASET_NAME} \
  --dataset_config_name=${DATASET_CONFIG_NAME} \
  --remove_unused_columns=False \
  --do_train \
  --do_eval \
  --dropout_prob=0.2 \
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
  --save_total_limit=2 \
  --report_to=wandb \
  --log_predictions \
  --load_best_model_at_end=True \
  --metric_for_best_model="eval_f1" \
  --fp16 \
  --half_precision_backend=apex \
  --fallback_fonts_dir=${FALLBACK_FONTS_DIR} \
  --seed=${SEED}
    # --do_predict \
