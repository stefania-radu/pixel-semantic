import matplotlib.pyplot as plt
from datasets import load_dataset
from PIL import Image
import numpy as np

from pixel import (
    AutoConfig,
    PangoCairoTextRenderer,
    PIXELForPreTraining,
    resize_model_embeddings,
    truncate_decoder_pos_embeddings,
    get_transforms,
)


config = AutoConfig.from_pretrained("Team-PIXEL/pixel-base")
text_renderer = PangoCairoTextRenderer.from_pretrained("configs/renderers/noto_renderer")
model = PIXELForPreTraining.from_pretrained("Team-PIXEL/pixel-base", config=config)

resize_model_embeddings(model, text_renderer.max_seq_length)
truncate_decoder_pos_embeddings(model, text_renderer.max_seq_length)

transforms = get_transforms(
    do_resize=True,
    size=(text_renderer.pixels_per_patch, text_renderer.pixels_per_patch * text_renderer.max_seq_length),
)


languages_dict = {
    "am": "Amharic",
    "ar": "Arabic",
    "fi": "Finnish",
    "ha": "Hausa",
    "id": "Indonesian",
    "ig": "Igbo",
    "ko": "Korean",
    "lg": "Luganda",
    "pcm": "Nigerian Pidgin",
    "rw": "Kinyarwanda",
    "sw": "Swahili",
    "te": "Telugu",
    "wo": "Wolof",
    "yo": "Yorùbá"
}

fig, axs = plt.subplots(2, 7, figsize=(20, 7))
axs = axs.flatten() 


for i, (lang_key, lang_value) in enumerate(languages_dict.items()):

    print(f"language: {lang_value}")
    
    # Load the dataset
    ds = load_dataset(f"stefania-radu/rendered_wikipedia_{lang_key}", split='train', streaming=True)

    ds_iter = iter(ds)
    
    sample = next(ds_iter)
    sample = next(ds_iter)
    sample = next(ds_iter)

    print(sample)
    print(sample["pixel_values"])

    img_array = np.array(sample["pixel_values"])
    print(f"shape: img_array = np.array(sample['pixel_values']) {img_array.shape}")
    
    image = transforms(Image.fromarray(img_array)).unsqueeze(0)
    print(f"shape: image = transforms(Image.fromarray(img_array)).unsqueeze(0) {image.shape}")
    
    original_img = model.unpatchify(model.patchify(image)).squeeze()
    print(f"shape: original_img = model.unpatchify(model.patchify(image)).squeeze() {original_img.shape}")

    num_patches = sample['num_patches']
    
    axs[i].imshow(original_img.permute(1,2,0))
    axs[i].axis('off')  # Hide axis for clarity
    axs[i].set_title(f"{lang_value.upper()} - Patches: {num_patches}", fontsize=12)

plt.tight_layout()

plt.savefig("rendered.pdf")
plt.show()
