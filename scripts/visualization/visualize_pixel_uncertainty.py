"""
Script that takes in an input string, renders it, masks out patches, and let's PIXEL reconstruct the image.
Adapted from ViT-MAE demo: https://github.com/facebookresearch/mae/blob/main/demo/mae_visualize.ipynb

Example usage:
python visualize_pixel_uncertainty.py \
  --input_str="After the release of ChatGPT in 2022, the number of papers published every day about Large Language
Models (LLMs) has increased more than 20-fold (Zhao et al., 2023). The number of parameters in these
LLMs jumped from 340 millions in implementations such as BERT (Devlin, Chang, Lee, & Toutanova,
2018) to billions of parameters in models like GPT-3 (Brown et al., 2020) or LLaMA (Touvron et al.,
2023). A large part of these parameters come from the word-embedding layers which are used to repre-
sent a finite vocabulary of characters, sets of characters or words. Apart from increasing model complex-
ity, a fixed vocabulary is also responsible for brittle models, which cannot deal with out-of-vocabulary
inputs and cannot cannot generalise to new languages. As a consequence, the performance in downstream
tasks is also affected." \
  --model_name_or_path="Team-PIXEL/pixel-base" \
  --span_mask \
  --mask_ratio=0.25 \
  --max_seq_length=256

"""

import argparse
import logging
import math
import sys

import torch
import wandb
from PIL import Image
from pixel import (
    AutoConfig,
    PIXELForPreTraining,
    PyGameTextRenderer,
    SpanMaskingGenerator,
    get_attention_mask,
    get_transforms,
    resize_model_embeddings,
    truncate_decoder_pos_embeddings,
)
from transformers import set_seed

logger = logging.getLogger(__name__)


def clip(x: torch.Tensor):
    x = torch.einsum("chw->hwc", x)
    x = torch.clip(x * 255, 0, 255)
    x = torch.einsum("hwc->chw", x)
    return x


def log_image(img: torch.Tensor, img_name: str, do_clip: bool = True):
    if do_clip:
        img = clip(img)
    wandb.log({img_name: wandb.Image(img)})


def main(args: argparse.Namespace):
    # Setup logging
    log_level = logging.INFO
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=log_level,
    )
    logger.setLevel(log_level)

    set_seed(args.seed)

    wandb.init()
    # wandb.run.name = args.revision

    config_kwargs = {
        "use_auth_token": args.auth_token if args.auth_token else None,
        "revision": args.revision,
    }

    # Load renderer
    text_renderer = PyGameTextRenderer.from_pretrained(
        args.renderer_name_or_path if args.renderer_name_or_path else args.model_name_or_path,
        max_seq_length=args.max_seq_length,
        **config_kwargs,
    )

    # Load model
    config = AutoConfig.from_pretrained(args.model_name_or_path, **config_kwargs)

    mask_ratio = args.mask_ratio if not args.manual_mask else len(args.manual_mask) / text_renderer.max_seq_length
    config.update({"mask_ratio": mask_ratio})

    model = PIXELForPreTraining.from_pretrained(args.model_name_or_path, config=config, **config_kwargs)

    # Resize position embeddings in case we use shorter sequence lengths
    resize_model_embeddings(model, text_renderer.max_seq_length)
    truncate_decoder_pos_embeddings(model, text_renderer.max_seq_length)

    logger.info("Running PIXEL masked autoencoding with pixel reconstruction")

    # Get transformations
    transforms = get_transforms(
        do_resize=True,
        size=(text_renderer.pixels_per_patch, text_renderer.pixels_per_patch * text_renderer.max_seq_length),
    )

    # Render input
    encoding = text_renderer(text=args.input_str)
    attention_mask = get_attention_mask(
        num_text_patches=encoding.num_text_patches, seq_length=text_renderer.max_seq_length
    )

    img = transforms(Image.fromarray(encoding.pixel_values)).unsqueeze(0)
    attention_mask = attention_mask.unsqueeze(0)
    inputs = {"pixel_values": img.float(), "attention_mask": attention_mask}

    # Generate the mask
    if args.manual_mask:
        logger.info("Using manual masking")
        mask = torch.zeros(size=(1, text_renderer.max_seq_length))
        for idx in args.manual_mask:
            mask[0][idx] = 1

    elif args.span_mask:
        mask_generator = SpanMaskingGenerator(
            num_patches=text_renderer.max_seq_length,
            num_masking_patches=math.ceil(mask_ratio * text_renderer.max_seq_length),
            max_span_length=args.masking_max_span_length,
            spacing=args.masking_spacing if args.masking_spacing else "span",
            cumulative_span_weights=args.masking_cumulative_span_weights,
        )
        logger.info(
            f'Applying span masking with "max_span_length = {args.masking_max_span_length}" '
            f', "cumulative_span_weights = {args.masking_cumulative_span_weights}" '
            f' and "spacing = {args.masking_spacing if args.masking_spacing else "span"}"'
        )
        mask = torch.tensor(mask_generator(num_text_patches=(encoding.num_text_patches + 1))).unsqueeze(0)
    else:
        logger.info("Using random masking")
        mask = None

    if mask is not None:
        masked_count = torch.count_nonzero(mask != 0, dim=1)[0]
        logger.info(f"Masked count: {masked_count}, ratio = {(masked_count / text_renderer.max_seq_length):0.4f}")
        inputs.update({"patch_mask": mask})
    else:
        logger.info(f"Masked count: {math.ceil(mask_ratio * text_renderer.max_seq_length)}, ratio = {mask_ratio:0.2f}")

    num_samples = 100  # Number of Monte Carlo samples
    all_predictions = []

    model.train()  # Activate dropout
    logger.info(f"Training mode: {model.training}") 
    for _ in range(num_samples):
        with torch.inference_mode():  # Disable gradient computation
            outputs = model(**inputs)
            predictions = model.unpatchify(outputs["logits"]).detach().cpu().squeeze()
            all_predictions.append(predictions)

    # Convert list of outputs to a tensor
    all_predictions = torch.stack(all_predictions)

    # Calculate mean and standard deviation
    mean_predictions = all_predictions.mean(dim=0)
    std_predictions = all_predictions.std(dim=0)
    var_predictions = all_predictions.var(dim=0)

    # visualize the mask
    mask = outputs["mask"].detach().cpu()

    # Log mask
    mask = mask.unsqueeze(-1).repeat(1, 1, text_renderer.pixels_per_patch ** 2 * 3)
    mask = model.unpatchify(mask).squeeze()  # 1 is removing, 0 is keeping
    log_image(mask, "mask")

    # Log attention mask
    attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, text_renderer.pixels_per_patch ** 2 * 3)
    attention_mask = model.unpatchify(attention_mask).squeeze()
    log_image(attention_mask, "attention_mask")

    # Log original image
    original_img = model.unpatchify(model.patchify(img)).squeeze()
    log_image(original_img, "original")

    # Log masked image
    im_masked = original_img * (1 - mask)
    log_image(im_masked, "masked")

    # Logging for Monte Carlo Dropout-based uncertainty
    log_image(mean_predictions, "mean_predictions", do_clip=False)
    log_image(var_predictions, "var_predictions", do_clip=False)
    log_image(mean_predictions - std_predictions, "mean_minus_std_predictions", do_clip=False)
    log_image(mean_predictions + std_predictions, "mean_plus_std_predictions", do_clip=False)

    # Generate and log a confidence map (inverse of std)
    confidence_map = 1 / (std_predictions + 1e-6)  # Adding a small value to avoid division by zero

    logger.info(confidence_map)
    logger.info(confidence_map.shape)
    logger.info(original_img.shape)
    logger.info(mean_predictions.shape)
    
    import matplotlib.pyplot as plt

    # Plot the confidence map with the defined colormap
    plt.imshow(confidence_map.mean(dim=0), vmin=0, vmax=255, cmap="Greens")
    plt.colorbar()
    plt.savefig('confidence_map.png')
    log_image("confidence_map.png", "confidence_map", do_clip=False)
    
    # masked_predictions = mean_predictions * mask * confidence_map
    # log_image(masked_predictions, "masked_predictions", do_clip=False)

    # # Assuming confidence_map is a 2D tensor
    # confidence_map_normalized = (confidence_map - confidence_map.min()) / (confidence_map.max() - confidence_map.min())
    # # Convert masked_predictions to RGB if it's single-channel
    # if masked_predictions.dim() == 2 or (masked_predictions.dim() == 3 and masked_predictions.shape[0] == 1):
    #     masked_predictions_rgb = torch.stack([masked_predictions] * 3, dim=0)  # Stack along the channel dimension
    # else:
    #     masked_predictions_rgb = masked_predictions

    # # Create a green overlay
    # confidence_map_normalized = confidence_map_normalized.unsqueeze(0) 
    # green_overlay = torch.zeros_like(masked_predictions_rgb)
    # green_overlay[1] = confidence_map_normalized[0]  # Modulate the green channel

    # alpha = 0.5  # Transparency factor for the overlay
    # combined_image = (1 - alpha) * masked_predictions_rgb + alpha * green_overlay

    # log_image(combined_image, "masked_predictions_with_confidence", do_clip=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_str", type=str, help="Path to already-rendered img or the raw string to encode")
    parser.add_argument("--model_name_or_path", type=str, help="Path to pretrained model")
    parser.add_argument("--renderer_name_or_path", type=str, default=None, help="Path to pretrained renderer")
    parser.add_argument("--auth_token", type=str, default="", help="HuggingFace auth token")
    parser.add_argument("--revision", type=str, default="main", help="HuggingFace branch name / commit ID")
    parser.add_argument("--max_seq_length", type=int, default=529, help="Maximum sequence length in patches")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--mask_ratio", type=float, default=0.25, help="Percentage of pixels that will be masked")
    parser.add_argument("--span_mask", action="store_true", help="Apply span masking")
    parser.add_argument(
        "--masking_max_span_length",
        type=int,
        default=6,
        help="Maximum span length that can be masked " "when using span masking",
    )
    parser.add_argument(
        "--masking_spacing",
        default=None,
        type=int,
        help="Spacing between masked spans. Defaults to the length of the span."
        "Use this argument to set it to a fixed number of patches."
        "Recommended setting: For masking ratio <= 0.4 leave the default"
        "For ratios between 0.4 and 0.7 set it to 1. For higher, set it to 0",
    )
    parser.add_argument(
        "--masking_cumulative_span_weights",
        type=str,
        default="0.2,0.4,0.6,0.8,0.9,1",
        help="Comma-separated list of cumulative probabilities of sampling a span of length n"
        "when using span masking. Must be a list of size model_args.masking_max_span_length.",
    )
    parser.add_argument("--manual_mask", nargs="+", type=int, default=None, help="Patch indices that should be masked")
    parsed_args = parser.parse_args()

    if parsed_args.masking_cumulative_span_weights:
        parsed_args.masking_cumulative_span_weights = [
            float(w) for w in parsed_args.masking_cumulative_span_weights.replace(" ", "").split(",")
        ]

    main(parsed_args)
