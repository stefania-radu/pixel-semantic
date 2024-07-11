#!/bin/bash
#SBATCH --time=3-00:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --job-name=GLUE-sst2-102
#SBATCH --output=GLUE-results/sst2/GLUE-sst2-102.out

# Note on GLUE: 
# We found that for some of the tasks (e.g. MNLI), PIXEL can get stuck in a bad local optimum
# A clear indicator of this is when the training loss is not decreasing substantially within the first 1k-3k steps
# If this happens, you can tweak the learning rate slightly, increase the batch size,
# change rendering backends, or often even just the random seed
# We are still trying to find the optimal training recipe for PIXEL on these tasks,
# the recipes used in the paper may not be the best ones out there

# Note on GLUE: 
# We found that for some of the tasks (e.g. MNLI), PIXEL can get stuck in a bad local optimum
# A clear indicator of this is when the training loss is not decreasing substantially within the first 1k-3k steps
# If this happens, you can tweak the learning rate slightly, increase the batch size,
# change rendering backends, or often even just the random seed
# We are still trying to find the optimal training recipe for PIXEL on these tasks,
# the recipes used in the paper may not be the best ones out there

module purge
module load Anaconda3/2023.03-1

conda activate pixel-sem-env2

# Optional wandb environment vars
export WANDB_PROJECT="pixel-experiments-SC"
export WANDB__SERVICE_WAIT=300

# Settings
export TASK="sst2"
export MODEL="Team-PIXEL/pixel-base" # also works with "bert-base-cased", "roberta-base", etc.
export RENDERING_BACKEND="pygame"  # Consider trying out both "pygame" and "pangocairo" to see which one works best
export POOLING_MODE="max" # Can be "mean", "max", "cls", or "pma1" to "pma8"
export SEQ_LEN=256
export BSZ=64 # it was 64 but it does not run on my machine
export GRAD_ACCUM=4  # We found that higher batch sizes can sometimes make training more stable
export LR=3e-5
export SEED=102
export NUM_STEPS=20000
  
export RUN_NAME="${TASK}-$(basename ${MODEL})-${POOLING_MODE}-${RENDERING_BACKEND}-${SEQ_LEN}-${BSZ}-${GRAD_ACCUM}-${LR}-${NUM_STEPS}-${SEED}"

python scripts/training/run_glue.py \
  --model_name_or_path=${MODEL} \
  --task_name=${TASK} \
  --rendering_backend=${RENDERING_BACKEND} \
  --pooling_mode=${POOLING_MODE} \
  --pooler_add_layer_norm=True \
  --remove_unused_columns=False \
  --do_train \
  --do_eval=False \
  --dropout_prob=0.2 \
  --max_seq_length=${SEQ_LEN} \
  --max_steps=${NUM_STEPS} \
  --num_train_epochs=10 \
  --early_stopping=False \
  --early_stopping_patience=5 \
  --per_device_train_batch_size=${BSZ} \
  --gradient_accumulation_steps=${GRAD_ACCUM} \
  --learning_rate=${LR} \
  --warmup_steps=100 \
  --run_name=${RUN_NAME} \
  --output_dir="GLUE-results/${TASK}/${RUN_NAME}" \
  --overwrite_output_dir \
  --overwrite_cache \
  --logging_strategy=steps \
  --logging_steps=100 \
  --evaluation_strategy="no" \
  --save_strategy="no" \
  --save_steps=${NUM_STEPS} \
  --save_total_limit=2 \
  --report_to=wandb \
  --log_predictions \
  --load_best_model_at_end=True \
  --metric_for_best_model="eval_accuracy" \
  --half_precision_backend=apex \
  --seed=${SEED} \
  --fp16 \
   
    
