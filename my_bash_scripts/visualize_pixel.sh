#!/bin/bash
#SBATCH --time=0-01:00:00
#SBATCH --partition=regular
#SBATCH --nodes=1
#SBATCH --mem=10GB
#SBATCH --job-name=visualize_pixel_attention
#SBATCH --output=visualize_pixel_attention.out


module purge
module load Anaconda3/2023.09-0

conda activate pixel-sem-env2

# Example usage:
python scripts/visualization/visualize_pixel_uncertainty.py\
  --input_str="""After the release of ChatGPT in 2022, the number of papers published every day about Large Language Models (LLMs) has increased more than 20-fold. The number of parameters in these LLMs jumped from 340 millions in implementations such as BERT to billions of parameters in models like GPT-3 or LLaMA. A large part of these parameters come from the word-embedding layers which are used to represent a finite vocabulary of characters, sets of characters or words. Apart from increasing model complexity, a fixed vocabulary is also responsible for brittle models, which cannot deal with out-of-vocabulary inputs and cannot generalize to new languages. As a consequence, the performance in downstream tasks is also affected. """ \
  --model_name_or_path="Team-PIXEL/pixel-base" \
  --span_mask \
  --mask_ratio=0.35 \
  --max_seq_length=256 \
  # --rgb=True