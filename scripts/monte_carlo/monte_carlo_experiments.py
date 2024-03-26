"""
Example usage:
python scripts\monte_carlo\monte_carlo_experiments.py \
    --model_name_or_path="Team-PIXEL/pixel-base"\
    --experiment_type="mask_ratio" \
    --do_loss \
    --do_std \
    --do_attention \
    --mask_ratio=0.25 \
    --masking_max_span_length=6 \
    --masking_cumulative_span_weights="0.2,0.4,0.6,0.8,0.9,1"\
    --span_mask\
    --max_seq_length=256

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
import json
import random
import torchvision.utils as vutils
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
file_handler = logging.FileHandler('std_outputs_mask_0.1.txt')
logger.addHandler(file_handler)

random.seed(42)


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

    num_patches_x = width // patch_size
    num_patches_y = height // patch_size

    all_heads_attentions = torch.zeros(nr_heads, height, width) # Adding channel dimension

    for head_idx, head in enumerate(attentions):
        attention_map = torch.zeros(height, width)
                
        for i in range(num_patches_y):
            for j in range(num_patches_x):
                if mask[:, i*patch_size, j*patch_size].any() == 0:
                    attention_map[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = head[i, j]
                # else:
                #     attention_map[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = 1

        all_heads_attentions[head_idx] = attention_map 

    # logger.info(f"all_heads_attentions.shape: {all_heads_attentions.shape}")

    return all_heads_attentions


def get_attention_grid(attention_tensor):

    if len(attention_tensor.shape) != 4:
        raise ValueError("Input tensor must be 4-dimensional [layers, heads, pixels, pixels].")

    layers, heads, pixels, _ = attention_tensor.shape

    # [layers*heads, 1, pixels, pixels] for grayscale images
    attention_tensor_reshaped = attention_tensor.view(layers*heads, 1, pixels, pixels)

    # Set the number of images in each row to the number of heads
    # each row corresponds to a layer
    grid = vutils.make_grid(attention_tensor_reshaped, nrow=heads, padding=2, normalize=True, scale_each=True)

    return grid


def save_grid(grid, id_text_attention, layers, heads):

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
    
    ax.set_xticklabels(range(1, heads + 1))
    ax.set_yticklabels(range(1, layers + 1))

    ax.set_xlabel("Layers", fontsize=20)
    ax.set_ylabel("Heads", fontsize=20)

    fig.colorbar(im, ax=ax)

    img_name = f"{id_text_attention}_grid.pdf"

    plt.savefig(img_name, bbox_inches='tight')
    
    plt.close()


def normalize_array(np_image):
    return (np_image - np.min(np_image)) / (np.max(np_image) - np.min(np_image))


def save_attention_image(all_layers_attentions, id_text_attention, layer_index, head_index):

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

    img_name = f'{id_text_attention}_attention_image_{layer_index+1}_{head_index+1}'
    # Save the figure
    plt.savefig(f'{img_name}.pdf', bbox_inches='tight')

    # log_image_from_path(f'{img_name}.pdf', img_name)
    
    plt.close()


def save_image(image_tensor, title, img_name): # shape: (3, 256, 256)
    
    image_tensor = image_tensor.mean(dim=0, keepdim=True)
    np_image = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)

    # np_image = normalize_array(np_image) # do not normalize

    plt.figure(figsize=(10, 10))
    img = plt.imshow(np_image, cmap='viridis')
    # plt.colorbar(img)
    # plt.title(title, fontsize=20)
    plt.axis('off') 
    plt.savefig(f'{img_name}.pdf', bbox_inches='tight')
    
    plt.close()


def save_multi_image(id_text, images, titles, final_img_name, mask_rate=0.1):
    assert len(images) == 3 and len(titles) == 3, "There must be exactly 3 images and 3 titles"

    np_images = [img.mean(dim=0).detach().cpu().numpy() for img in images]

    global_min = min(img.min() for img in np_images)
    global_max = max(img.max() for img in np_images)

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    for i, (np_image, title) in enumerate(zip(np_images, titles)):
        np_image_normalized = (np_image - global_min) / (global_max - global_min)

        im = axs[i].imshow(np_image_normalized, cmap='viridis')
        axs[i].set_title(title, fontsize=24)
        axs[i].axis('off')

    plt.tight_layout(pad=2.0)

    fig.suptitle(f'Mask Span = {mask_rate} - ID: {id_text}', fontsize=26, y=1.1)  # for mask span experiment

    cbar_ax = fig.add_axes([axs[-1].get_position().x1 + 0.01, axs[-1].get_position().y0, 0.02, axs[-1].get_position().height])
    
    fig.colorbar(im, cax=cbar_ax)

    plt.savefig(f'{final_img_name}.pdf', bbox_inches='tight')
    plt.close()


def dump_data(args, data_dict, name_start):
    
    if args.experiment_type == "mask_ratio":
            name_end = f"mask_{args.mask_ratio}"
    elif args.experiment_type == "span":
        name_end = f"span_{args.masking_max_span_length}"

    name = f'{name_start}_{name_end}.json'

    logger.info(f"Saving data in {name}")
        
    with open(name, 'w') as f:
        json.dump(data_dict, f, ensure_ascii=False, indent=4)


def load_loss(args):

    if args.experiment_type == "mask_ratio":
        name = f"loss_per_task_mask_{args.mask_ratio}.json"
    elif args.experiment_type == "span":
        name = f"loss_per_task_span_{args.masking_max_span_length}.json"

    logger.info(f"Loading data from {name}")
    
    with open(name, 'r', encoding='utf-8') as f:
        loss_scores = json.load(f)

    return loss_scores


def compute_attention(all_attentions, id_text_attention, mask, text_renderer, rows_to_save=[1, 10], cols_to_save=[2, 4]):

    logger.info(f"Computing attention for ID: {id_text_attention}")
    
    all_attentions_mean = all_attentions.mean(dim=0).squeeze(0)

    # make the attention weights grid plot
    all_layers_attentions = []
    for layer in all_attentions_mean:
        all_heads_attentions_image = visualize_attention(layer, mask, text_renderer.pixels_per_patch)
        all_layers_attentions.append(all_heads_attentions_image)

    all_layers_attentions = torch.stack(all_layers_attentions)

    logger.info(f"all_layers_attentions: {all_layers_attentions.shape}")

    attention_grid = get_attention_grid(all_layers_attentions)
    attention_grid = torch.mean(attention_grid, dim=0, keepdim=True)

    logger.info(f"attention_grid: {attention_grid.shape}")

    # save attention weights grid with all layers and heads - all channels are the same
    # save_grid(attention_grid[0:1, :, :], layers=all_layers_attentions.size(0), heads=all_layers_attentions.size(1))

    save_grid(attention_grid, id_text_attention, layers=all_layers_attentions.size(0), heads=all_layers_attentions.size(1))

    # save example images from the grid at given layer and head
    for row, col in zip(rows_to_save, cols_to_save):
        save_attention_image(all_layers_attentions, id_text_attention, row, col)


def monte_carlo_loss(args, data, model, text_renderer):

    num_samples = 100  # Number of Monte Carlo samples
    logger.info(f"Monte Carlo samples: {num_samples}")

    model.train()  # Activate dropout
    logger.info(f"Training mode: {model.training}") 

    losses_per_task = data.copy()

    for task, lang_dict in data.items():
        logger.info(f"\n######## Computing Loss for task: {task} ########")
        
        for lang, id_dict in lang_dict.items():
            logger.info(f"\n######## Language: {lang} ######## \n")
            
            for id_text, text in id_dict.items():
                
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
                        num_masking_patches=math.ceil(args.mask_ratio * text_renderer.max_seq_length),
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
                    logger.info(f"Masked count: {math.ceil(args.mask_ratio * text_renderer.max_seq_length)}, ratio = {args.mask_ratio:0.2f}")


                logger.info(f"\nID text: {id_text}")
                all_losses = []

                for _ in range(num_samples):
                    with torch.inference_mode():  # Disable gradient computation
                        outputs = model(**inputs)
                        loss = outputs["loss"].detach().cpu()
                        all_losses.append(loss)

                all_losses = torch.stack(all_losses)
                mean_loss = all_losses.mean(dim=0)
                logger.info(f"mean loss: {mean_loss.item()}")
                
                losses_per_task[task][lang][id_text] = mean_loss.item()

    logger.info(f"Losses per task: {losses_per_task}")

    # Save losses 
    dump_data(args, losses_per_task, "loss_per_task")
    
    return losses_per_task


def monte_carlo_SD(args, input_data, model, text_renderer, mask_ratio, extreme_losses_dict):

    num_samples = 100  # Number of Monte Carlo samples
    logger.info(f"Monte Carlo samples: {num_samples}")
    
    model.train()  # Activate dropout
    logger.info(f"Training mode: {model.training}") 

    SDs_per_task = input_data.copy()
    var_per_task = input_data.copy()

    lowest_id = next(iter(extreme_losses_dict["lowest"]), None)
    highest_id = next(iter(extreme_losses_dict["highest"]), None)

    for task, lang_dict in input_data.items():
        logger.info(f"\n######## Computing SDs for task: {task} ########\n")
        
        for lang, id_dict in lang_dict.items():
            logger.info(f"\n######## Language: {lang} ######## \n")
            
            for id_text, text in id_dict.items():

                in_attention = (id_text == lowest_id or id_text == highest_id)

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
                
                all_predictions = []
                all_attentions = []

                for _ in range(num_samples):
                    with torch.inference_mode():  # Disable gradient computation
                        outputs = model(**inputs)
                        
                        predictions = model.unpatchify(outputs["logits"]).detach().cpu().squeeze() # (batch_size, patch_size ** 2 * num_channels)
                        all_predictions.append(predictions)

                        if in_attention:
                            id_text_attention = id_text
                            attentions = torch.cat(outputs["attentions"])
                            all_attentions.append(attentions)

                # Convert list of outputs to a tensor
                all_predictions = torch.stack(all_predictions)
                if all_attentions:
                    all_attentions = torch.stack(all_attentions)

                # Calculate mean, variance and standard deviation, dim0 = 100
                mean_predictions = all_predictions.mean(dim=0)
                std_predictions = all_predictions.std(dim=0)
                var_predictions = all_predictions.var(dim=0)

                # mask
                mask = outputs["mask"].detach().cpu()  
                mask = mask.unsqueeze(-1).repeat(1, 1, text_renderer.pixels_per_patch ** 2 * 3)
                mask = model.unpatchify(mask).squeeze()  # 1 is removing, 0 is keeping

                # original image
                original_img = model.unpatchify(model.patchify(img)).squeeze()

                # masked image
                im_masked = original_img * mask * attention_mask

                # mean variance per image
                mean_variance_value = var_predictions.mean().item()
                logger.info(f"Mean variance for {id_text}: {mean_variance_value}")

                # mean SD per image
                mean_std_value = std_predictions.mean().item()
                logger.info(f"Mean std for for {id_text}: {mean_std_value}")

                SDs_per_task[task][lang][id_text] = mean_std_value
                var_per_task[task][lang][id_text] = mean_variance_value

                # compute std per each patch
                std_predictions_per_patch = create_mean_map(std_predictions, mask, text_renderer.pixels_per_patch) # black is 0,  torch.Size([3, 368, 368])
                
                # Original with SD-per-patch on top
                std_predictions_per_patch_with_original = original_img * (std_predictions_per_patch) * attention_mask# for var I had 1 - mean_var
                
                # Mean predictions with SD-per-patch on top
                std_reconstruction_per_patch = clip(mean_predictions * std_predictions_per_patch * mask * attention_mask)

                if args.do_attention and in_attention:
                    compute_attention(all_attentions, id_text_attention, mask, text_renderer, rows_to_save=[1, 10], cols_to_save=[2, 4])

                # save plots for the 5 images with the lowest and highest losses
                if id_text in extreme_losses_dict["low"] or id_text in extreme_losses_dict["high"]:
    
                    # save individual images: original, original+SD, predicitons+SD
                    save_image(im_masked, title=f'Original \n {id_text}', img_name=f"{id_text}_original")
                    save_image(std_predictions_per_patch_with_original, title=f'Original + Patch Uncertainty (SD) \n {id_text}', img_name=f"{id_text}_original_SD_m{args.mask_ratio}")
                    save_image(std_reconstruction_per_patch, title=f'Mean Prediction + SD \n {id_text}', img_name=f"{id_text}_predictions_SD_m{args.mask_ratio}")

                    # make triplets of images
                    images = [im_masked, std_predictions_per_patch_with_original, std_reconstruction_per_patch]
                    titles = [f"Original", "Original + SD", "Mean Prediction + SD"]
                    multi_image_name = f"{id_text}_triplet_m{args.mask_ratio}"

                    save_multi_image(id_text, images, titles, multi_image_name, args.mask_ratio)
            

    logger.info(f"SD per task: {SDs_per_task}")

    # Save SD 
    dump_data(args, SDs_per_task, "SD_per_task")
    dump_data(args, mean_variance_value, "var_per_task")
        
    return SDs_per_task


def find_extreme_loss_ids(losses_per_task, value=5):

    losses_dict = {}
    all_losses = []
    for task, langs in losses_per_task.items():
        for lang, id_losses in langs.items():
            all_losses.extend(id_losses.items())
    
    sorted_losses = sorted(all_losses, key=lambda x: x[1])

    lowest_losses_tuples = sorted_losses[:value]
    losses_dict["low"] = {id_text: loss for id_text, loss in lowest_losses_tuples}
    
    highest_losses_tuples = sorted_losses[-value:]
    losses_dict["high"] = {id_text: loss for id_text, loss in highest_losses_tuples}

    losses_dict["lowest"] = {sorted_losses[0][0]: sorted_losses[0][1]}
    losses_dict["highest"] = {sorted_losses[-1][0]: sorted_losses[-1][1]}

    logger.info("Lowest losses:")
    for id_text, loss in losses_dict["low"].items():
        logger.info(f"ID: {id_text}, Loss: {loss}")

    logger.info("Highest losses:")
    for id_text, loss in losses_dict["high"].items():
        logger.info(f"ID: {id_text}, Loss: {loss}")
    
    return losses_dict


def compute_means_per_task(losses_per_task):
    mean_losses_per_task = {}

    for task, langs in losses_per_task.items():
        mean_losses_per_task[task] = {}
        for lang, id_losses in langs.items():
            losses = torch.tensor(list(id_losses.values()))
            # Compute the mean loss for the current language
            mean_loss = losses.mean().item()
            mean_losses_per_task[task][lang] = mean_loss

    logger.info(f"Mean loss per task: {mean_losses_per_task}")
    
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

    logger.info(f"Running MONTE CARLO experiment: {args.experiment_type}")

    input_data_path = r"scripts/data/uncertainty/test_data_ner_tydiqa_glue_small.json"
    # input_data_path = r"scripts\data\uncertainty\test_data_ner_tydiqa_glue_small.json"

    # get small dataset with 10 examples per language/subtask
    with open(input_data_path, 'r', encoding='utf-8') as f:
        input_data = json.load(f)

    if args.do_loss:
        loss_scores = monte_carlo_loss(args, input_data, model, text_renderer)
    else:
        loss_scores = load_loss(args)

    logger.info("\nLOSS\n")
    compute_means_per_task(loss_scores)

    # find the 5 entries with the lowest loss and 5 with highest
    extreme_losses_dict = find_extreme_loss_ids(loss_scores, 5) # keys: low, high, lowest, highest

    if args.do_std:

        # maybe this will fix the float error and I can run both loss and std flags at the same time
        with open(input_data_path, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
            
        SD_scores = monte_carlo_SD(args, input_data, model, text_renderer, mask_ratio, extreme_losses_dict)

        logger.info("\nUNCERTAINTY (SD)\n")
        compute_means_per_task(SD_scores)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, help="Path to pretrained model")
    parser.add_argument("--renderer_name_or_path", type=str, default=None, help="Path to pretrained renderer")
    parser.add_argument("--auth_token", type=str, default="", help="HuggingFace auth token")
    parser.add_argument("--revision", type=str, default="main", help="HuggingFace branch name / commit ID")
    parser.add_argument("--max_seq_length", type=int, default=529, help="Maximum sequence length in patches")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--mask_ratio", type=float, default=0.25, help="Percentage of pixels that will be masked")
    parser.add_argument("--span_mask", action="store_true", help="Apply span masking")
    parser.add_argument("--rgb", action="store_true", help="Apply span masking")
    parser.add_argument("--do_loss", action="store_true", help="Compute and save the loss scores for all examples. When false, loss scores are loaded.")
    parser.add_argument("--do_std", action="store_true", help="Compute and save the SD scores usign Monte Carlo.")
    parser.add_argument("--do_attention", action="store_true", help="Compute and save attention grid for the input")
    parser.add_argument("--experiment_type", type=str, default="mask_ratio", help="Set type of experiment: 'mask_ratio' or 'span'")
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
