# Script that takes in an input string, renders it, masks out patches, and let's PIXEL reconstruct the image.
# Adapted from ViT-MAE demo: https://github.com/facebookresearch/mae/blob/main/demo/mae_visualize.ipynb

# Example usage:
python scripts/visualization/visualize_pixel.py \
  --input_str="I am writing my Master thesis on the PIXEL model." \
  --model_name_or_path="Team-PIXEL/pixel-base" \
  --span_mask \
  --mask_ratio=0.25 \
  --max_seq_length=256