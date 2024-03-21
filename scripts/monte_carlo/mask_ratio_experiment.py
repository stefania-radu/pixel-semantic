# Re-importing necessary libraries after reset
import os
import json
import matplotlib.pyplot as plt
import numpy as np

def extract_mask_ratio(filename):
    """Extract mask ratio from file name."""
    try:
        basename = os.path.basename(filename)
        mask_ratio_str = basename.split('_')[-1].split('.')[0]
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

def plot_loss_vs_mask_ratio(folder_path, across='tasks'):
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
                    for task in data.values():
                        for lang, lang_data in task.items():
                            if lang not in losses:
                                losses[lang] = []
                            mean_loss = calculate_mean_loss({"temp_task": {lang: lang_data}})
                            losses[lang].append(mean_loss)
                
                mask_ratios.append(mask_ratio)
    
    # Sort data by mask ratio
    sorted_indices = np.argsort(mask_ratios)
    mask_ratios = np.array(mask_ratios)[sorted_indices]

    # Plotting
    plt.figure(figsize=(10, 6))
    for key, vals in losses.items():
        sorted_vals = np.array(vals)[sorted_indices]
        plt.plot(mask_ratios, sorted_vals, label=key)

    plt.xlabel('Mask Ratio')
    plt.ylabel('Mean Loss')
    plt.title('Mean Loss vs. Mask Ratio')
    plt.legend()
    plt.show()

# Note: Update the 'folder_path' variable with the actual path to the folder containing your JSON files before running.
# plot_loss_vs_mask_ratio('path_to_your_folder', across='tasks')
# plot_loss_vs_mask_ratio('path_to_your_folder', across='languages')
