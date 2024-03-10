"""
Script that takes in an input string, renders it, masks out patches, and let's PIXEL reconstruct the image.
Adapted from ViT-MAE demo: https://github.com/facebookresearch/mae/blob/main/demo/mae_visualize.ipynb

Example usage:
python scripts/visualization/visualize_pixel_uncertainty.py\
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
  --max_seq_length=256 \

"""

import argparse
import logging
import math
import sys
import numpy as np

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
from pdf2image import convert_from_path
import torchvision.utils as vutils
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def clip(x: torch.Tensor):
    x = torch.einsum("chw->hwc", x)
    x = torch.clip(x * 255, 0, 255)
    x = torch.einsum("hwc->chw", x)
    return x


def log_image(img: torch.Tensor, img_name: str, do_clip: bool = True, mode=None):
    if do_clip:
        img = clip(img)
    wandb.log({img_name: wandb.Image(img)})


def log_image_from_path(img_path: str, img_name: str):
    if img_path.endswith('.pdf'):
        images = convert_from_path(img_path)
        img = images[0]  # Assuming there is only one page in the PDF
    else:
        img = Image.open(img_path)

    wandb.log({img_name: wandb.Image(img)})


def create_mean_map(variance_image, mask, patch_size):
    """
    Create a map where each value in a patch equals the mean of values in that patch,
    only for patches where the mask is 1. Other patches are set to black.

    Parameters:
    variance_image (torch.Tensor): The original 3-channel variance image. It can also be the std image
    mask (np.array or torch.Tensor): A binary mask indicating which patches to process.
    patch_size (int): The size of each square patch.

    Returns:
    np.array: Mean variance map.
    """
    # Average across channels
    # if len(variance_image.shape) == 3 and variance_image.shape[0] == 3:
    #     variance_image = variance_image.mean(dim=0)

    # Convert to numpy if they are tensors
    if isinstance(variance_image, torch.Tensor):
        variance_image = variance_image.numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy()

    num_channels, height, width = variance_image.shape

    # Initialize the mean variance map
    mean_variance_map = np.zeros_like(variance_image)

    # Calculate the number of patches in each dimension
    num_patches_x = width // patch_size
    num_patches_y = height // patch_size

    # iterate over channels
    for c in range(num_channels):
        # Iterate over each patch
        for i in range(num_patches_y):
            for j in range(num_patches_x):
                # Check the corresponding value in the mask
                if mask[0][i*patch_size, j*patch_size] != 0:
                    # Extract the patch
                    patch = variance_image[c, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
                    
                    # Calculate the mean variance of the patch
                    mean_variance = np.mean(patch)

                    # Assign this mean variance to all pixels in the patch
                    mean_variance_map[c, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = mean_variance
                else:
                    # Set the patch to black if mask is 0
                    mean_variance_map[c, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = 0

    return mean_variance_map


def visualize_attention(attentions, mask, patch_size):
    attentions = attentions.detach().cpu().squeeze()

    nr_heads, _, _ = attentions.shape
    num_channels, height, width = mask.shape
    print(f"mask.shape: {mask.shape}")
    print(f"attentions.shape: {attentions.shape}")

    # Calculate the number of patches in each dimension
    num_patches_x = width // patch_size
    num_patches_y = height // patch_size

    # Initialize a tensor to store all attention maps
    all_heads_attentions = torch.zeros(nr_heads, height, width) # Adding channel dimension

    for head_idx, head in enumerate(attentions):
        attention_map = torch.zeros(height, width)
        
        num_nonzero_elements = (head != 0).sum()
        # print(f"Number of elements different from 0 in head: {num_nonzero_elements}")
                
        for i in range(num_patches_y):
            for j in range(num_patches_x):
                if mask[:, i*patch_size, j*patch_size].any() == 0:
                    attention_map[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = head[i, j]
                # else:
                #     attention_map[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = 1

        all_heads_attentions[head_idx] = attention_map 

    print(f"all_heads_attentions.shape: {all_heads_attentions.shape}")

    return all_heads_attentions


def get_attention_grid(attention_tensor):
    """
    Combine all attention maps into a single image grid.

    Parameters:
    - attention_tensor: A 4D tensor with shape [layers, heads, pixels, pixels].

    Returns:
    - A single image tensor representing the grid of all attention maps.
    """
    # Validate input tensor shape
    if len(attention_tensor.shape) != 4:
        raise ValueError("Input tensor must be 4-dimensional [layers, heads, pixels, pixels].")

    layers, heads, pixels, _ = attention_tensor.shape

    # Reshape the tensor to treat layers and heads as separate batches,
    # this way, each attention map is treated as an individual image
    # [layers*heads, 1, pixels, pixels] for grayscale images
    attention_tensor_reshaped = attention_tensor.view(layers*heads, 1, pixels, pixels)

    # Use torchvision's make_grid to combine these images into a grid
    # Set the number of images in each row to the number of heads
    # This creates a grid where each row corresponds to a layer
    grid = vutils.make_grid(attention_tensor_reshaped, nrow=heads, padding=2, normalize=True, scale_each=True)

    return grid

def save_grid(grid, layers, heads):
    """
    Save the attention grid with labels for axes to a PDF file without re-normalizing the whole grid.

    Parameters:
    - grid: The image tensor representing the grid of all attention maps.
    - layers: The number of layers in the attention tensor.
    - heads: The number of heads in the attention tensor.
    """
    # Convert grid to numpy array
    np_image = grid.detach().cpu().numpy().transpose(1, 2, 0)
    if np_image.shape[-1] == 1:  # For grayscale images, if there's a single channel
        np_image = np_image.squeeze(-1)

    fig, ax = plt.subplots(figsize=(12, 12))
    im = ax.imshow(np_image, cmap='viridis')

    # Adjusted calculation for ticks based on the single height (H) and width (W) of the grid image
    xticks_positions = [i * np_image.shape[1] / heads + (np_image.shape[1] / heads / 2) for i in range(heads)]
    yticks_positions = [i * np_image.shape[0] / layers + (np_image.shape[0] / layers / 2) for i in range(layers)]

    ax.set_xticks(xticks_positions)
    ax.set_yticks(yticks_positions)
    ax.set_xlabel("Layers", fontsize=16)
    ax.set_ylabel("Heads", fontsize=16)
    ax.set_xticklabels(range(1, heads + 1))
    ax.set_yticklabels(range(1, layers + 1))

    ax.set_xlabel("Layers", fontsize=20)
    ax.set_ylabel("Heads", fontsize=20)

    fig.colorbar(im, ax=ax)  # Optional: Adds a colorbar

    img_name = "grid"

    plt.savefig(f"{img_name}.pdf", bbox_inches='tight')
    log_image_from_path(f"{img_name}.pdf", img_name)
    
    plt.close()


def normalize_array(np_image):
    return (np_image - np.min(np_image)) / (np.max(np_image) - np.min(np_image))


def save_attention_image(all_layers_attentions, layer_index, head_index):
    """
    Save a single attention map image from the specified layer and head with the 'viridis' colormap.

    Parameters:
    - all_layers_attentions: A 4D tensor of shape [layers, heads, height, width].
    - layer_index: The index of the layer to visualize.
    - head_index: The index of the head to visualize.
    - filename: The name of the file to save the image to.
    """
    # Validate indices
    layers, heads, _, _ = all_layers_attentions.shape
    if layer_index >= layers or head_index >= heads:
        raise ValueError("Layer index or head index is out of bounds.")
    
    # Extract the specific attention map
    attention_map = all_layers_attentions[layer_index, head_index, :, :].detach().cpu().numpy()

    # Normalize attention map between 0 and 1
    attention_map = normalize_array(attention_map)
    
    # Plotting
    plt.figure(figsize=(10, 10))
    img = plt.imshow(attention_map, cmap='viridis')
    plt.colorbar(img)
    plt.title(f'Layer {layer_index + 1}, Head {head_index + 1}', fontsize=20)
    plt.axis('off')  # Optionally, turn off the axis for a cleaner image

    img_name = f'attention_image_{layer_index+1}_{head_index+1}'
    # Save the figure
    plt.savefig(f'{img_name}.pdf', bbox_inches='tight')

    log_image_from_path(f'{img_name}.pdf', img_name)
    
    plt.close()


def save_image(image_tensor, title, img_name): # shape: (3, 256, 256)
    image_tensor = image_tensor.mean(dim=0, keepdim=True)
    np_image = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)

    np_image = normalize_array(np_image)

    # Plotting
    plt.figure(figsize=(10, 10))
    img = plt.imshow(np_image, cmap='viridis')
    # plt.colorbar(img)
    # plt.title(title, fontsize=20)
    plt.axis('off')  # Optionally, turn off the axis for a cleaner image

    # Save the figure
    plt.savefig(f'{img_name}.pdf', bbox_inches='tight')

    log_image_from_path(f'{img_name}.pdf', img_name)
    
    plt.close()


def save_multi_image(images, titles, final_img_name, mask_rate=0.1):  # shapes: [(3, 256, 256), (3, 256, 256), (3, 256, 256)]
    assert len(images) == 3 and len(titles) == 3, "There must be exactly 3 images and 3 titles"

    # Create a figure and a set of subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))  # Adjusted for equal size images including colorbar space

    for i, (image_tensor, title) in enumerate(zip(images, titles)):
        # Convert the PyTorch tensor to a NumPy array and normalize
        image_tensor = image_tensor.mean(dim=0, keepdim=True)  # Convert to grayscale for demonstration
        np_image = image_tensor.detach().cpu().numpy().squeeze()  # Assuming the input is PyTorch tensor
        # np_image = (np_image - np_image.min()) / (np_image.max() - np_image.min())  # Normalize

        # Plotting each image in its subplot
        im = axs[i].imshow(np_image, cmap='viridis')
        axs[i].set_title(title, fontsize=24)
        axs[i].axis('off')  # Optionally, turn off the axis for a cleaner image

    # Adjust layout to be tight and allocate space for colorbar
    plt.tight_layout(pad=2.0)

    # fig.suptitle(f'Mask Ratio = {int(mask_rate*100)}%', fontsize=26, y=1.1) # for mask ratio experiment
    fig.suptitle(f'Mask Span = {mask_rate}', fontsize=26, y=1.1) # for mask span experiment

    # Create an axes on the right side of axs[-1]. The width of cax can be controlled by the horizontal size, here set to 0.015
    cbar_ax = fig.add_axes([axs[-1].get_position().x1 + 0.01, axs[-1].get_position().y0, 0.02, axs[-1].get_position().height])
    
    # Add colorbar to the newly created axis
    fig.colorbar(im, cax=cbar_ax)

    # Save the figure
    plt.savefig(f'{final_img_name}.pdf', bbox_inches='tight')
    plt.close()



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

    # wandb.init()
    # wandb.run.name = args.revision

    # experiments_table = wandb.Table(columns=["input_text", "MC_samples", "mean_std", "mean_variance", "patches_count", "masked_ratio", "masked_count", "max_span_length", "cumulative_span_weights"])

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

    # wandb.log({"mask_ratio": mask_ratio})

    model = PIXELForPreTraining.from_pretrained(args.model_name_or_path, config=config, **config_kwargs)

    dict_config = model.config.to_dict()

    # hyperparams_table = wandb.Table(columns=list(dict_config.keys()), data=[list(dict_config.values())])
    # wandb.log({"hyperparams_table": hyperparams_table})

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
    logger.info(f"Monte Carlo samples: {num_samples}")
    all_predictions = []
    all_attentions = []

    model.train()  # Activate dropout
    logger.info(f"Training mode: {model.training}") 
    for _ in range(num_samples):
        with torch.inference_mode():  # Disable gradient computation
            outputs = model(**inputs)
            predictions = model.unpatchify(outputs["logits"]).detach().cpu().squeeze() # (batch_size, patch_size ** 2 * num_channels)
            # attentions = torch.cat(outputs["attentions"])
            all_predictions.append(predictions)
            # all_attentions.append(attentions)

    # Convert list of outputs to a tensor
    all_predictions = torch.stack(all_predictions)
    # all_attentions = torch.stack(all_attentions)

    # print(f"all_attention (samples, layers, batch_size, num_heads, sequence_length, sequence_length): {all_attentions.shape}")

    # all_attentions_mean = all_attentions.mean(dim=0).squeeze(0)
        
    # print(f"all_attention after mean: {all_attentions_mean.shape}")

    # Calculate mean and standard deviation
    mean_predictions = all_predictions.mean(dim=0)
    std_predictions = all_predictions.std(dim=0)
    var_predictions = all_predictions.var(dim=0)

    logger.info(f"std_predictions shape: {std_predictions.shape}")

    # Log mask
    mask = outputs["mask"].detach().cpu()  
    mask = mask.unsqueeze(-1).repeat(1, 1, text_renderer.pixels_per_patch ** 2 * 3)
    mask = model.unpatchify(mask).squeeze()  # 1 is removing, 0 is keeping
    # log_image(mask, "mask")

    ########## ATTENTION STUFF #################

    # make the attention weights grid plot
    # all_layers_attentions = []
    # for layer_idx, layer in enumerate(all_attentions_mean):
    #     all_heads_attentions_image = visualize_attention(layer, mask, text_renderer.pixels_per_patch)
    #     all_layers_attentions.append(all_heads_attentions_image)
    #     print(f"all_heads_attentions_image: {all_heads_attentions_image.shape}")

    # all_layers_attentions = torch.stack(all_layers_attentions)

    # print(f"all_layers_attentions: {all_layers_attentions.shape}")

    # # save example images from the grid at given layer and head
    # save_attention_image(all_layers_attentions, 1, 2)
    # save_attention_image(all_layers_attentions, 10, 4)

    # attention_grid = get_attention_grid(all_layers_attentions)
    # print(f"attention_grid: {attention_grid.shape}")
    # print(f"attention_grid 0 : {attention_grid[0]}")
    # print(f"attention_grid 1 : {attention_grid[1]}")
    # are_different = (attention_grid[0] != attention_grid[1]).any()
    # print(f"Are the channels different? {are_different}")

    # # save attention weights grid with all layers and heads - all channels are the same
    # save_grid(attention_grid[0:1, :, :], layers=all_layers_attentions.size(0), heads=all_layers_attentions.size(1))
    # log_image(attention_grid, "attention_grid", do_clip=False)

    ########## ATTENTION STUFF #################
    

    # Log attention mask - where the renderer is looking
    attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, text_renderer.pixels_per_patch ** 2 * 3)
    attention_mask = model.unpatchify(attention_mask).squeeze()
    # log_image(attention_mask, "attention_mask")
    # logger.info(f"attention_mask: {attention_mask}")
    
    # Log original image
    original_img = model.unpatchify(model.patchify(img)).squeeze()
    # save_image(original_img, title=f'Original', img_name=f"original")

    # Log masked image
    im_masked = original_img * (1 - mask)
    # log_image(im_masked, "masked")


    mean_variance_value = np.round(var_predictions.mean().item(), 3)
    # mean_variance_value = np.round(var_predictions.mean(dim=0).mean(), 3)
    logger.info(f"Mean variance for whole image: {mean_variance_value}")
    # wandb.log({"mean_variance_value": mean_variance_value})


    # Compute mean std per whole image
    mean_std_value = np.round(std_predictions.mean().item(), 3)
    # mean_std_value = np.round(std_predictions.mean(dim=0).mean(), 3)
    logger.info(f"Mean std for whole image: {mean_std_value}")
    # wandb.log({"mean_std_value": mean_std_value})

    # compute std per each patch and log the new map
    std_predictions_per_patch = create_mean_map(std_predictions, mask, text_renderer.pixels_per_patch) # black is 0,  torch.Size([3, 368, 368])
    logger.info(f"SD image: {std_predictions_per_patch[0]}")
    mean_std_value_patch_mean = np.round(std_predictions_per_patch.mean().item(), 3)
    logger.info(f"Mean std for whole image patch mean: {mean_std_value_patch_mean}")
    # wandb.log({"mean_std_value_patch_mean": mean_std_value_patch_mean})
    
    logger.info(f"mean_std shape: {std_predictions_per_patch.shape}")
    std_predictions_per_patch_with_original = original_img * (std_predictions_per_patch) # for var I had 1 - mean_var
    logger.info(f"std_predictions shape: {std_predictions_per_patch_with_original.shape}")
    # log_image(std_predictions_per_patch_with_original, "std_predictions_patch", do_clip=False)
    save_image(std_predictions_per_patch_with_original, title=f'Original + Patch Uncertainty (SD)', img_name=f"original_SD_s{args.masking_max_span_length}")

    # Log just the std image, without per patch mean
    # std_predictions_per_pixel = std_predictions
    # std_predictions_per_pixel_with_original = original_img * (std_predictions_per_pixel)
    # log_image(std_predictions_per_pixel_with_original, "std_predictions_pixel", do_clip=False)


    # log reconstructed image with per patch std
    std_reconstruction_per_patch = clip(mean_predictions * std_predictions_per_patch * mask)
    log_image(std_reconstruction_per_patch, "std_reconstruction_per_patch", do_clip=False)
    
    print(f"mean_predictions: {mean_predictions}")
    print(f"std_predictions_per_patch: {std_reconstruction_per_patch[0][0]}")
    print(f"std_reconstruction_per_patch: {std_reconstruction_per_patch[0][0]}")
    
    save_image(std_reconstruction_per_patch, title=f'Mean Prediction + SD', img_name=f"predictions_SD_s{args.masking_max_span_length}")

    # log reconstructed image with per pixel std
    # std_reconstruction_per_pixel = mean_predictions * (std_predictions_per_pixel)
    # log_image(std_reconstruction_per_pixel, "std_reconstruction_pixel", do_clip=False)


    ##############################
    #make triplets of images
    images = [im_masked, std_predictions_per_patch_with_original, std_reconstruction_per_patch]
    titles = ["Original", "Original + SD", "Mean Prediction + SD"]
    multi_image_name = f"triplet_s{args.masking_max_span_length}"

    save_multi_image(images, titles, multi_image_name, args.masking_max_span_length)
    ##############################


    # experiments_table.add_data(args.input_str, num_samples, mean_std_value, mean_variance_value, std_predictions.shape[1], mask_ratio, masked_count, args.masking_max_span_length, args.masking_cumulative_span_weights)
    # wandb.log({"experiments_table": experiments_table})

    logger.info(original_img.shape)
    logger.info(mean_predictions.shape)
       


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
    parser.add_argument("--rgb", action="store_true", help="Apply span masking")
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
