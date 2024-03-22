import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pdf2image import convert_from_path

# Function to parse the loss info file and return lists of IDs for lowest and highest losses
def parse_loss_file(filepath):
    with open(filepath, 'r') as file:
        content = file.read()
    sections = content.split('Highest losses:')
    lowest = sections[0].split('\n')[1:-1]
    highest = sections[1].split('\n')[1:]
    
    lowest_ids = [line.split(', ')[0].split(': ')[1] for line in lowest]
    highest_ids = [line.split(', ')[0].split(': ')[1] for line in highest]
    
    return lowest_ids, highest_ids

# Helper function to convert PDF to image
def convert_pdf_to_img(pdf_path):
    images = convert_from_path(pdf_path)
    return images[0]

# Function to fetch and plot images for given IDs
def plot_images_for_ids(ids, folders, output_file):
    images_per_id = []
    titles = ['Original', 'Original + SD', 'Predictions + SD']  # Column titles
    for id_ in ids:
        id_images = []
        for folder in folders:
            for file in os.listdir(folder):
                if id_ in file:
                    img_path = os.path.join(folder, file)
                    if img_path.lower().endswith('.pdf'):
                        # Convert PDF to image if it's a PDF file
                        img = convert_pdf_to_img(img_path)
                    else:
                        # This is a fallback for non-PDF files, not expected in this use case
                        img = Image.open(img_path)
                    id_images.append(img)
                    break  # Assumes only one relevant image per folder
        images_per_id.append(id_images)
    
    # Normalize images
    all_images = [np.array(img) for sublist in images_per_id for img in sublist]
    min_val = min([img.min() for img in all_images])
    max_val = max([img.max() for img in all_images])
    
    fig, axs = plt.subplots(len(ids), 3, figsize=(10, len(ids)*3.3))
    for i, ax in enumerate(axs[0]):
        ax.set_title(titles[i])
    for ax_row, id_images in zip(axs, images_per_id):
        for ax, img in zip(ax_row, id_images):
            ax.imshow(np.array(img), cmap='viridis', vmin=min_val, vmax=max_val)
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file, format='pdf')

# Define your paths and file
loss_info_file = 'scripts/monte_carlo/results/base experiment/lowest_highest_loss_info.txt'
folders = [
    'scripts/monte_carlo/results/base experiment/images/original',
    'scripts/monte_carlo/results/base experiment/images/original_SD',
    'scripts/monte_carlo/results/base experiment/images/predictions_SD'
]

# Parse the loss info file
lowest_ids, highest_ids = parse_loss_file(loss_info_file)

# Plot images for the IDs with the lowest losses
plot_images_for_ids(lowest_ids, folders, 'lowest_losses_plot.pdf')

# Plot images for the IDs with the highest losses
plot_images_for_ids(highest_ids, folders, 'highest_losses_plot.pdf')
