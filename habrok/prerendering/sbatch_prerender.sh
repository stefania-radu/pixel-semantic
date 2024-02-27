#!/bin/bash
#SBATCH --time=3-00:00:00
#SBATCH --partition=himem
#SBATCH --nodes=1
#SBATCH --mem=3900GB
#SBATCH --job-name=prerender_wikipedia_multilingual_en
#SBATCH --output=prerender_wikipedia_multilingual_en.out


module purge
module load Anaconda3/2023.03-1

conda activate pixel-sem-env2

# pip install -U datasets huggingface-hub

# languages: [am, ha, ig, rw, lg, pcm, sw, wo, yo, ar, fi, id, ru, te]

export RENDERER_PATH="configs/renderers/noto_renderer"
export LANG="en"

python scripts/data/prerendering/prerender_wikipedia_multilingual.py \
  --lang=${LANG} \
  --renderer_name_or_path=${RENDERER_PATH} \
  --chunk_size=10000000000 \
  --repo_id="stefania-radu/rendered_wikipedia_${LANG}" \
  --split="train" \
  --auth_token="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"