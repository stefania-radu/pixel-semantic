import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pdf2image import convert_from_path
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

def parse_loss_file(filepath):
    with open(filepath, 'r') as file:
        content = file.read()
    sections = content.split('Highest losses:')
    lowest = sections[0].split('\n')[1:-1]
    highest = sections[1].split('\n')[1:]
    
    lowest_ids = [line.split(', ')[0].split(': ')[1] for line in lowest]
    highest_ids = [line.split(', ')[0].split(': ')[1] for line in highest]
    
    return lowest_ids, highest_ids

def convert_pdf_to_img(pdf_path):
    images = convert_from_path(pdf_path)
    return images[0]

def plot_images_for_ids(ids, folders, output_file, plot_title, languages):
    images_per_id = []
    titles = ['Original', 'Original + SD', 'Predictions + SD']
    
    for id_ in ids:
        id_images = []
        for folder in folders:
            for file in os.listdir(folder):
                if id_ in file:
                    img_path = os.path.join(folder, file)
                    if img_path.lower().endswith('.pdf'):
                        img = convert_pdf_to_img(img_path)
                    else:
                        img = Image.open(img_path)
                    id_images.append(img)
                    break
        images_per_id.append(id_images)
    
    all_images = [np.array(img) for sublist in images_per_id for img in sublist]
    min_val = min([img.min() for img in all_images])
    max_val = max([img.max() for img in all_images])
    
    fig, axs = plt.subplots(len(ids), 4, figsize=(8, 2.5 * len(ids)), gridspec_kw={'width_ratios': [0.5, 4.8, 4.8, 6]})
    fig.suptitle(plot_title, fontsize=14)
    
    for ax in axs.flat:
        ax.axis('off')

    for i, title in enumerate(titles):
        axs[0, i+1].set_title(title, fontsize=13)

    for row, (lang, id_images) in enumerate(zip(languages, images_per_id)):
        axs[row, 0].text(1, 0.5, lang, va='center', ha='center', rotation=90, fontsize=13)
        for col, img in enumerate(id_images, start=1):
            im = axs[row, col].imshow(np.array(img), cmap='viridis', vmin=min_val, vmax=max_val)
    
    plt.tight_layout(rect=[0, 0.03, 0.95, 0.97])  # Adjust layout to make room for colorbar

    # Create a fake ScalarMappable for the colorbar
    sm = ScalarMappable(cmap='viridis', norm=Normalize(vmin=min_val / 250, vmax=max_val / 250))
    sm.set_array([])
    # Add colorbar to the right side of the plot
    cbar = fig.colorbar(sm, ax=axs[:, -1], location='right', aspect=50)

    plt.savefig(output_file, format='pdf')


# Define your paths and file
loss_info_file = 'scripts/monte_carlo/results/base_experiment/lowest_highest_loss_info.txt'
folders = [
    'scripts/monte_carlo/results/base_experiment/images/original',
    'scripts/monte_carlo/results/base_experiment/images/original_SD',
    'scripts/monte_carlo/results/base_experiment/images/predictions_SD'
]

lowest_ids, highest_ids = parse_loss_file(loss_info_file)

lowest_losses_languages = ["English", "English", "English", "English", "English"]
highest_losses_languages = ["Korean", "Korean", "English", "Chinese", "Korean"]

plot_images_for_ids(lowest_ids, folders, 'lowest_losses_plot.pdf', "Top 5 Performers", lowest_losses_languages)
plot_images_for_ids(highest_ids, folders, 'highest_losses_plot.pdf', "Top 5 Challenges", highest_losses_languages)

