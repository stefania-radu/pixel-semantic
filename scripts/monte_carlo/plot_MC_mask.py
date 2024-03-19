import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pdf2image import convert_from_path


def plot_MC_mask_experiment(original_path, original_sd_path, predictions_sd_path):
     # Convert the original PDF image to PNG
    original_images = convert_from_path(original_path)
    original_image = original_images[0].convert('L')  # Assume the first page and convert to grayscale
    
    def convert_and_load_image(path):
        images = convert_from_path(path)
        return images[0].convert('L')  # Assume the first page and convert to grayscale
    
    original_sd_files = sorted(os.listdir(original_sd_path), key=lambda x: float(x.split('_s')[1].replace('.pdf', ''))) # # change to m for mask experiment
    predictions_sd_files = sorted(os.listdir(predictions_sd_path), key=lambda x: float(x.split('_s')[1].replace('.pdf', ''))) # change to m for mask experiment
    
    N = len(original_sd_files)
    
    fig = plt.figure(figsize=(16, 5*N))
    gs = gridspec.GridSpec(N, 5, width_ratios=[0.05, 1, 1, 1, 0.05], wspace=0.05, hspace=0.05)

    for i, (orig_sd_file, pred_sd_file) in enumerate(zip(original_sd_files, predictions_sd_files)):
        mask_rate = float(orig_sd_file.split('_s')[1].replace('.pdf', '')) # change to m for mask experiment
        
        # Text for mask rate
        text_ax = plt.subplot(gs[i, 0])
        # text_ax.text(0.5, 0.5, f'span length = {int(mask_rate*100)}%', va='center', ha='center', rotation=90, fontsize=22)
        text_ax.text(0.5, 0.5, f'span length = {int(mask_rate)}', va='center', ha='center', rotation=90, fontsize=22)
        text_ax.axis('off')
        
        orig_sd_img = convert_and_load_image(os.path.join(original_sd_path, orig_sd_file))
        pred_sd_img = convert_and_load_image(os.path.join(predictions_sd_path, pred_sd_file))
        
        # Original image
        ax1 = plt.subplot(gs[i, 1])
        im = ax1.imshow(np.array(original_image), cmap='viridis')
        ax1.axis('off')
        
        # Original + SD image
        ax2 = plt.subplot(gs[i, 2])
        ax2.imshow(np.array(orig_sd_img), cmap='viridis')
        ax2.axis('off')
        
        # Mean Predictions + SD image
        ax3 = plt.subplot(gs[i, 3])
        ax3.imshow(np.array(pred_sd_img), cmap='viridis')
        ax3.axis('off')

    # Titles
    plt.subplot(gs[0, 1]).set_title('Original', fontsize=24)
    plt.subplot(gs[0, 2]).set_title('Original + SD', fontsize=24)
    plt.subplot(gs[0, 3]).set_title('Mean Predictions + SD', fontsize=24)
    
    # Colorbar
    cbar_ax = plt.subplot(gs[:, -1])
    plt.colorbar(im, cax=cbar_ax)
    plt.savefig('span_comparison_with_colorbar.pdf', bbox_inches='tight')


def plot_line_graph(experiments_path):
    experiments_table = pd.read_csv(experiments_path)

    # Group the data by masked_ratio and calculate the mean of mean_std for each group
    grouped_data = experiments_table.groupby('max_span_length')['mean_std'].mean().reset_index()

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(grouped_data['max_span_length'], grouped_data['mean_std'], marker='o', linestyle='-', color='b')
    # plt.title('Mean Patch SD vs. Mask Ratio', fontsize=24)
    plt.xlabel('Span Length', fontsize=22)
    plt.ylabel('Mean Patch SD', fontsize=22)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig('std_vs_span.pdf', bbox_inches='tight')
    

original_path = "results_plots/MC_uncertainty/mask_ratio_experiment/original.pdf"
original_sd_path = "results_plots/MC_uncertainty/mask_span_experiment/original_SD"
predictions_sd_path = "results_plots/MC_uncertainty/mask_span_experiment/predictions_SD"
experiments_path = "results_plots/MC_uncertainty/mask_span_experiment/experiments.csv"

# plot the mask ratio comparision between image triplets
plot_MC_mask_experiment(original_path, original_sd_path, predictions_sd_path)

# plot std vs mask ratio line graph
plot_line_graph(experiments_path)