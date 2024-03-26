import os
import json
import matplotlib.pyplot as plt
import numpy as np

def extract_mask_ratio(filename):
    """Extract mask ratio from file name."""
    try:
        basename = os.path.basename(filename)
        mask_ratio_str = os.path.splitext(basename)[0].split('_')[-1]
        return float(mask_ratio_str)
    except ValueError:
        return None

def calculate_mean_loss(data):
    """Calculate the mean loss from a nested dictionary."""
    total_loss = 0
    count = 0
    for task in data.values():
        for language in task.values():
            for loss in language.values():
                total_loss += loss
                count += 1
    return total_loss / count if count > 0 else 0

def plot_loss_vs_mask_ratio(folder_path, across='tasks', what="loss_mask"):
    """Plot loss versus mask ratio."""
    # Initialize data storage
    mask_ratios = []
    losses = {'ner': [], 'tydiqa': [], 'glue': []} if across == 'tasks' else {}

    # Iterate over files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            mask_ratio = extract_mask_ratio(filename)
            if mask_ratio is not None:
                with open(os.path.join(folder_path, filename), 'r') as f:
                    data = json.load(f)
                
                if across == 'tasks':
                    for task in losses:
                        if task in data:
                            mean_loss = calculate_mean_loss({task: data[task]})
                            losses[task].append(mean_loss)
                else:  # across languages
                    for task, lang_data in data.items():
                        for lang, text_data in lang_data.items():
                            # Adjusting labels for English and GLUE tasks
                            # if lang == 'english' or lang == 'conll_2003_en' or task == 'glue':
                            #     adjusted_lang = 'English'
                            # else:
                            adjusted_lang = lang
                            if adjusted_lang not in losses:
                                losses[adjusted_lang] = []
                            mean_loss = calculate_mean_loss({"temp_task": {adjusted_lang: text_data}})
                            losses[adjusted_lang].append(mean_loss)
                
                mask_ratios.append(mask_ratio)
    
    # Sort data by mask ratio
    sorted_indices = np.argsort(mask_ratios)
    mask_ratios = np.array(mask_ratios)[sorted_indices]

    # Plotting
    plt.figure(figsize=(20, 10))
    for key, vals in losses.items():
        sorted_vals = np.array(vals)[sorted_indices]
        plt.plot(mask_ratios, sorted_vals, label=key)

    plt.xlabel('Mask Ratio', fontsize=16)
    plt.ylabel('Mean Loss', fontsize=16)
    plt.legend(title='Task' if across == 'tasks' else 'Language', fontsize=14, loc="best")
    plt.grid(True)

    img_name = f"{what}_{across}_line_plot.pdf"

    plt.savefig(img_name, bbox_inches='tight')

# Note: Update the 'folder_path' variable with the actual path to the folder containing your JSON files before running.

path_to_folder = "scripts/monte_carlo/results/mask_experiment/loss_scores"
plot_loss_vs_mask_ratio(path_to_folder, across='tasks', what="loss_mask")
plot_loss_vs_mask_ratio(path_to_folder, across='languages', what="loss_mask")
