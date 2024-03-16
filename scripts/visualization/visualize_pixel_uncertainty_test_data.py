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
    PangoCairoTextRenderer,
    SpanMaskingGenerator,
    get_attention_mask,
    get_transforms,
    resize_model_embeddings,
    truncate_decoder_pos_embeddings,
)
from transformers import set_seed
from pdf2image import convert_from_path
import json
import random
import torchvision.utils as vutils
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
random.seed(42)

dict_tasks = {"ner": {"amh": {}, 
                      "conll": {}, 
                      "hau": {}, 
                      "ibo": {}, 
                      "kin": {}, 
                      "lug": {}, 
                      "luo": {},
                      "pcm": {}, 
                      "swa": {}, 
                      "wol": {}, 
                      "yor": {},
                      "zh": {}},
              "tydiqa": {"arabic": {}, 
                         "telugu": {}, 
                         "swahili": {}, 
                         "japanese": {}, 
                         "finnish": {},
                         "indonesian": {}, 
                         "russian": {}, 
                         "thai": {}, 
                         "korean": {}, 
                         "bengali": {}, 
                         "english": {}},
              "glue": {"cola": {}, 
                       "mnli": {}, 
                       "mrpc": {}, 
                       "qnli": {}, 
                       "qqp": {}, 
                       "rte": {}, 
                       "sst2": {}, 
                       "stsb": {}, 
                       "wnli": {}}} # make this a dict where the values are tasks


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

    # log_image_from_path(f'{img_name}.pdf', img_name)
    
    plt.close()


def save_image(image_tensor, title, img_name): # shape: (3, 256, 256)
    image_tensor = image_tensor.mean(dim=0, keepdim=True)
    np_image = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)

    # np_image = normalize_array(np_image) # do not normalize

    # Plotting
    plt.figure(figsize=(10, 10))
    img = plt.imshow(np_image, cmap='viridis')
    # plt.colorbar(img)
    # plt.title(title, fontsize=20)
    plt.axis('off')  # Optionally, turn off the axis for a cleaner image

    # Save the figure
    plt.savefig(f'{img_name}.pdf', bbox_inches='tight')

    # log_image_from_path(f'{img_name}.pdf', img_name)
    
    plt.close()


def save_multi_image(id_text, images, titles, final_img_name, mask_rate=0.1):
    assert len(images) == 3 and len(titles) == 3, "There must be exactly 3 images and 3 titles"

    # Convert all images to NumPy arrays and to grayscale
    np_images = [img.mean(dim=0).detach().cpu().numpy() for img in images]  # List of grayscale images as NumPy arrays

    # Compute global min and max across all images
    global_min = min(img.min() for img in np_images)
    global_max = max(img.max() for img in np_images)

    # Create a figure and a set of subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))  # Adjusted for equal size images including colorbar space

    for i, (np_image, title) in enumerate(zip(np_images, titles)):
        # Normalize current image using global min and max
        np_image_normalized = (np_image - global_min) / (global_max - global_min)

        # Plotting each image in its subplot
        im = axs[i].imshow(np_image_normalized, cmap='viridis')
        axs[i].set_title(title, fontsize=24)
        axs[i].axis('off')  # Optionally, turn off the axis for a cleaner image

    # Adjust layout to be tight and allocate space for colorbar
    plt.tight_layout(pad=2.0)

    fig.suptitle(f'Mask Span = {mask_rate} - ID: {id_text}', fontsize=26, y=1.1)  # for mask span experiment

    # Create an axes on the right side of axs[-1]. The width of cax can be controlled by the horizontal size, here set to 0.015
    cbar_ax = fig.add_axes([axs[-1].get_position().x1 + 0.01, axs[-1].get_position().y0, 0.02, axs[-1].get_position().height])
    
    # Add colorbar to the newly created axis
    fig.colorbar(im, cax=cbar_ax)

    # Save the figure
    plt.savefig(f'{final_img_name}.pdf', bbox_inches='tight')
    plt.close()


def get_losses(args, data, text_renderer, mask_ratio, model):

    losses_per_task = {"ner": {}, "tydiqa": {}, "glue": {}}

    for task, value in data.items():  # task, {id:text}
        for id_text, text in value.items():
            lang = id_text.split("-")[0] if task == "tydiqa" else id_text.split("_")[0]

            if lang not in losses_per_task[task]:
                losses_per_task[task][lang] = {}  # Initialize if this language is not yet in the dict

            
            # Get transformations
            transforms = get_transforms(
                do_resize=True,
                size=(text_renderer.pixels_per_patch, text_renderer.pixels_per_patch * text_renderer.max_seq_length),
            )

            # Render input
            encoding = text_renderer(text=text)
            attention_mask = get_attention_mask(
                num_text_patches=encoding.num_text_patches, seq_length=text_renderer.max_seq_length
            )

            img = transforms(Image.fromarray(encoding.pixel_values)).unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            inputs = {"pixel_values": img.float(), "attention_mask": attention_mask}

            # attention of renderer
            attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, text_renderer.pixels_per_patch ** 2 * 3)
            attention_mask = model.unpatchify(attention_mask).squeeze()

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


            logger.info(f"ID text: {id_text}")

            num_samples = 100  # Number of Monte Carlo samples
            logger.info(f"Monte Carlo samples: {num_samples}")
            all_losses = []

            model.train()  # Activate dropout
            logger.info(f"Training mode: {model.training}") 
            for _ in range(num_samples):
                with torch.inference_mode():  # Disable gradient computation
                    outputs = model(**inputs)
                    loss = outputs["loss"].detach().cpu()
                    all_losses.append(loss)

            all_losses = torch.stack(all_losses)
            mean_loss = all_losses.mean(dim=0)
            logger.info(f"mean loss: {mean_loss.item()}")
            
            losses_per_task[task][lang][id_text] = mean_loss.item()

    logger.info(f"losses per task: {losses_per_task}")
    return losses_per_task


def do_monte_carlo(args, id_text, text, text_renderer, model, mask_ratio, save=False):

    # Get transformations
    transforms = get_transforms(
        do_resize=True,
        size=(text_renderer.pixels_per_patch, text_renderer.pixels_per_patch * text_renderer.max_seq_length),
    )

    # Render input
    encoding = text_renderer(text=text)
    attention_mask = get_attention_mask(
        num_text_patches=encoding.num_text_patches, seq_length=text_renderer.max_seq_length
    )

    img = transforms(Image.fromarray(encoding.pixel_values)).unsqueeze(0)
    attention_mask = attention_mask.unsqueeze(0)
    inputs = {"pixel_values": img.float(), "attention_mask": attention_mask}

    # attention of renderer
    attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, text_renderer.pixels_per_patch ** 2 * 3)
    attention_mask = model.unpatchify(attention_mask).squeeze()

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


    # examples_to_show = ["glue_cola_en_0", "glue_cola_en_1", "bam_0", "fon_405", "arabic-2387335860751143628-1", "indonesian--7977769598018620690-0"]

    # logger.info(f"Examples to show: {examples_to_show}")
    logger.info(f"ID text: {id_text}")

    num_samples = 100  # Number of Monte Carlo samples
    logger.info(f"Monte Carlo samples: {num_samples}")
    all_predictions = []
    all_losses = []
    all_attentions = []

    model.train()  # Activate dropout
    logger.info(f"Training mode: {model.training}") 
    for _ in range(num_samples):
        with torch.inference_mode():  # Disable gradient computation
            outputs = model(**inputs)
            predictions = model.unpatchify(outputs["logits"]).detach().cpu().squeeze() # (batch_size, patch_size ** 2 * num_channels)
            loss = outputs["loss"].detach().cpu()
            # attentions = torch.cat(outputs["attentions"])
            all_predictions.append(predictions)
            all_losses.append(loss)
            # all_attentions.append(attentions)

    # Convert list of outputs to a tensor
    all_predictions = torch.stack(all_predictions)
    all_losses = torch.stack(all_losses)
    # all_attentions = torch.stack(all_attentions)

    # Calculate mean and standard deviation
    mean_predictions = all_predictions.mean(dim=0)
    std_predictions = all_predictions.std(dim=0)
    var_predictions = all_predictions.var(dim=0)

    mean_loss = all_losses.mean(dim=0)

    # logger.info(f"std_predictions shape: {std_predictions.shape}")

    # Log mask
    mask = outputs["mask"].detach().cpu()  
    mask = mask.unsqueeze(-1).repeat(1, 1, text_renderer.pixels_per_patch ** 2 * 3)
    mask = model.unpatchify(mask).squeeze()  # 1 is removing, 0 is keeping
    # log_image(mask, "mask")

     # Log original image
    original_img = model.unpatchify(model.patchify(img)).squeeze()

    # Log masked image
    im_masked = original_img * mask * attention_mask

    if save:
        save_image(im_masked, title=f'Original \n {id_text}', img_name=f"original_{id_text}")
    # log_image(im_masked, "masked")


    mean_variance_value = var_predictions.mean().item()
    # mean_variance_value = np.round(var_predictions.mean(dim=0).mean(), 3)
    logger.info(f"Mean variance for {id_text}: {mean_variance_value}")
    # wandb.log({"mean_variance_value": mean_variance_value})


    # Compute mean std per whole image
    mean_std_value = std_predictions.mean().item()
    # mean_std_value = np.round(std_predictions.mean(dim=0).mean(), 3)
    logger.info(f"Mean std for for {id_text}: {mean_std_value}")
    logger.info(f"Loss for {id_text}: {mean_loss}")
    # wandb.log({"mean_std_value": mean_std_value})

    # compute std per each patch and log the new map
    std_predictions_per_patch = create_mean_map(std_predictions, mask, text_renderer.pixels_per_patch) # black is 0,  torch.Size([3, 368, 368])
    # logger.info(f"SD image: {std_predictions_per_patch[0]}")
    mean_std_value_patch_mean = std_predictions_per_patch.mean().item()
    # logger.info(f"Mean std for whole image patch mean: {mean_std_value_patch_mean}")
    # wandb.log({"mean_std_value_patch_mean": mean_std_value_patch_mean})
    
    # logger.info(f"mean_std shape: {std_predictions_per_patch.shape}")
    std_predictions_per_patch_with_original = original_img * (std_predictions_per_patch) * attention_mask# for var I had 1 - mean_var
    # logger.info(f"std_predictions shape: {std_predictions_per_patch_with_original.shape}")
    # log_image(std_predictions_per_patch_with_original, "std_predictions_patch", do_clip=False)
    
    if save:
        save_image(std_predictions_per_patch_with_original, title=f'Original + Patch Uncertainty (SD) \n {id_text}', img_name=f"original_SD_m{args.mask_ratio}_{id_text}")
    
    # log reconstructed image with per patch std
    std_reconstruction_per_patch = clip(mean_predictions * std_predictions_per_patch * mask * attention_mask)
    # log_image(std_reconstruction_per_patch, "std_reconstruction_per_patch", do_clip=False)
    

    if save:
        save_image(std_reconstruction_per_patch, title=f'Mean Prediction + SD \n {id_text}', img_name=f"predictions_SD_m{args.mask_ratio}_{id_text}")

    ##############################
    #make triplets of images
    images = [im_masked, std_predictions_per_patch_with_original, std_reconstruction_per_patch]
    titles = [f"Original", "Original + SD", "Mean Prediction + SD"]
    multi_image_name = f"triplet_m{args.mask_ratio}_{id_text}"

    if save:
        save_multi_image(id_text, images, titles, multi_image_name, args.mask_ratio)
    ##############################

    return mean_std_value, mean_loss

def calculate_mean(scores):
    return sum(scores) / len(scores) if scores else 0


def find_extreme_loss_ids(losses_per_task, value=5):
    # Flatten the losses_per_task to a list of (id_text, loss) tuples
    all_losses = []
    for task, langs in losses_per_task.items():
        for lang, id_losses in langs.items():
            all_losses.extend(id_losses.items())
    
    # Sort by loss value
    sorted_losses = sorted(all_losses, key=lambda x: x[1])

    # Get the IDs with the lowest losses and convert to dictionary
    lowest_losses_tuples = sorted_losses[:value]
    lowest_losses_dict = {id_text: loss for id_text, loss in lowest_losses_tuples}
    
    # Get the IDs with the highest losses and convert to dictionary
    highest_losses_tuples = sorted_losses[-value:]
    highest_losses_dict = {id_text: loss for id_text, loss in highest_losses_tuples}
    
    return lowest_losses_dict, highest_losses_dict


def compute_means_per_task(losses_per_task):
    mean_losses_per_task = {}

    for task, langs in losses_per_task.items():
        mean_losses_per_task[task] = {}
        for lang, id_losses in langs.items():
            # Extract all losses for the current language as a list
            losses = torch.tensor(list(id_losses.values()))
            # Compute the mean loss for the current language
            mean_loss = losses.mean().item()
            # Assign the mean loss to the corresponding language under the current task
            mean_losses_per_task[task][lang] = mean_loss
            
    return mean_losses_per_task


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

    std_scores = dict_tasks.copy()
    # loss_scores = dict_tasks.copy()

    input_data = {}
    with open('data/test_data_semantic_tasks/test_data_for_rendering_masakhaner.json') as f:
        input_data["ner"] = json.load(f)

    with open('data/test_data_semantic_tasks/test_data_for_rendering_tydiqa.json') as f:
        input_data["tydiqa"] = json.load(f)
        
    with open('data/test_data_semantic_tasks/test_data_for_rendering_glue.json') as f:
        input_data["glue"] = json.load(f)

    random_data = {}
    random_data["ner"] = {k: v for k, v in random.sample(input_data["ner"].items(), 100)}
    random_data["tydiqa"] = {k: v for k, v in random.sample(input_data["tydiqa"].items(), 100)}
    random_data["glue"] = {k: v for k, v in random.sample(input_data["glue"].items(), 100)}

    losses = get_losses(args, random_data, text_renderer, mask_ratio, model)

    with open(f'losses_per_task_m{mask_ratio}.json', 'w') as f:
            json.dump(losses, f, ensure_ascii=False, indent=4)

    logger.info("\n LOSS \n")

    means_loss_per_task = compute_means_per_task(losses)
    logger.info(means_loss_per_task)

    lowest_losses_dict, highest_losses_dict = find_extreme_loss_ids(losses)

    logger.info("lowest_losses:")
    for id_text, loss in lowest_losses_dict.items():
        logger.info(f"ID: {id_text}, Loss: {loss}")

    logger.info("highest_losses:")
    for id_text, loss in highest_losses_dict.items():
        logger.info(f"ID: {id_text}, Loss: {loss}")


    highest_losses_dict = lowest_losses_dict.update(highest_losses_dict)
    # exit()

    for idx, (id_text, text) in enumerate(random_data["ner"].items()):
        save = True if idx in [0, 1] else False # save = True if idx in highest_losses_dict.keys() else False
        lang = id_text.split("_")[0]
        std_score = do_monte_carlo(args, id_text, text, text_renderer, model, mask_ratio, save=save)
        std_scores["ner"][lang][id_text] = std_score

    logger.info("DONE Ner")

    for idx, (id_text, text) in enumerate(random_data["tydiqa"].items()):
        save = True if idx in [0, 1] else False # # save = True if idx in highest_losses_dict.keys() else False
        lang = id_text.split("-")[0]
        std_score = do_monte_carlo(args, id_text, text, text_renderer, model, mask_ratio, save=save)
        std_scores["tydiqa"][lang][id_text] = std_score

    logger.info("DONE tydiqa")

    for idx, (id_text, text) in enumerate(random_data["glue"].items()):
        save = True if idx in [0, 1] else False # save = True if idx in highest_losses_dict.keys() else False
        task = id_text.split("_")[0]
        std_score = do_monte_carlo(args, id_text, text, text_renderer, model, mask_ratio, save=save)
        std_scores["glue"][lang][id_text] = std_score

    logger.info("DONE glue")

    logger.info("\n RESULTS \n")

    with open(f'std_scores_m{mask_ratio}.json', 'w') as f:
            json.dump(std_scores, f, ensure_ascii=False, indent=4)

    logger.info("\n SD \n")

    means_sd_per_task = compute_means_per_task(std_scores)
    logger.info(means_sd_per_task)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--input_str", type=str, help="Path to already-rendered img or the raw string to encode")
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
