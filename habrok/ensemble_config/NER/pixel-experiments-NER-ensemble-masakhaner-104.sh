#!/bin/bash
#SBATCH --time=0-05:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:2
#SBATCH --ntasks=10
#SBATCH --mem=80G
#SBATCH --job-name=pixel-experiments-NER-ensemble-104
#SBATCH --output=/home2/s3919609/pixel-semantic/NER-results/pixel-experiments-NER-ensemble-104.out


module purge
module load Anaconda3/2023.03-1

conda activate pixel-sem-env2

export WANDB_PROJECT="pixel-experiments-NER"


languages=(yor)

# Loop through each language
for LANG in "${languages[@]}"
do
  echo "Running job for language: $LANG"
  export LANG="$LANG"
  export DATA_DIR="/scratch/s3919609/data/masakhane-ner/data/${LANG}"
  export FALLBACK_FONTS_DIR="/scratch/s3919609/data/fallback_fonts"  # location of downloaded fonts
  export MODEL="Team-PIXEL/pixel-base" # can be switched with "bert-base-cased", "roberta-base", etc.
  export SEQ_LEN=196
  export BSZ=16
  export GRAD_ACCUM=1
  export LR=5e-5
  export SEED=104
  export NUM_STEPS=15000 
  
  export RUN_NAME="${LANG}-$(basename ${MODEL})-${SEQ_LEN}-${BSZ}-${GRAD_ACCUM}-${LR}-${NUM_STEPS}-${SEED}"

  python scripts/training/run_ner.py \
    --model_name_or_path=${MODEL} \
    --remove_unused_columns=False \
    --data_dir=${DATA_DIR} \
    --do_train \
    --do_eval \
    --do_predict \
    --dropout_prob=0.2 \
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
    --output_dir="NER-results/${LANG}/${RUN_NAME}" \
    --overwrite_output_dir \
    --overwrite_cache \
    --logging_strategy=steps \
    --logging_steps=500 \
    --evaluation_strategy=steps \
    --eval_steps=15000 \
    --save_strategy=steps \
    --save_steps=15000 \
    --save_total_limit=1 \
    --report_to=wandb \
    --log_predictions \
    --load_best_model_at_end=True \
    --metric_for_best_model="eval_f1" \
    --fp16 \
    --half_precision_backend=apex \
    --fallback_fonts_dir=${FALLBACK_FONTS_DIR} \
    --seed=${SEED}

done