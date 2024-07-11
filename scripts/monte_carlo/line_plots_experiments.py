import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set_style("darkgrid")

language_codes = {
    "amh": "Amharic",
    "conll_2003_en": "English",
    "hau": "Hausa",
    "ibo": "Igbo",
    "kin": "Kinyarwanda",
    "lug": "Luganda",
    "luo": "Luo",
    "pcm": "Nigerian Pidgin",
    "swa": "Swahili",
    "wol": "Wolof",
    "yor": "Yoruba",
    "zh": "Chinese",
    "arabic": "Arabic",
    "russian": "Russian",
    "bengali": "Bengali",
    "telugu": "Telugu",
    "finnish": "Finnish",
    "swahili": "Swahili",
    "korean": "Korean",
    "indonesian": "Indonesian",
    "english": "English",
    "cola": "English",
    "mnli": "English",
    "mrpc": "English",
    "qnli": "English",
    "qqp": "English",
    "rte": "English",
    "sst2": "English",
    "stsb": "English",
    "wnli": "English"
}

language_scripts = {
    "amh": "Ge'ez",
    "conll_2003_en": "Latin",
    "hau": "Latin",
    "ibo": "Latin",
    "kin": "Latin",
    "lug": "Latin",
    "luo": "Latin",
    "pcm": "Latin",
    "swa": "Latin",
    "wol": "Latin",
    "yor": "Latin",
    "zh": "Chinese characters",
    "arabic": "Arabic",
    "russian": "Cyrillic",
    "bengali": "Bengali",
    "telugu": "Telugu",
    "finnish": "Latin",
    "swahili": "Latin",
    "korean": "Korean",
    "indonesian": "Latin",
    "english": "Latin",
    "cola": "Latin",
    "mnli": "Latin",
    "mrpc": "Latin",
    "qnli": "Latin",
    "qqp": "Latin",
    "rte": "Latin",
    "sst2": "Latin",
    "stsb": "Latin",
    "wnli": "Latin"
}

task_codes = {"ner": "MasakhaNER",
              "tydiqa": "TyDiQA-GoldP",
              "glue": "GLUE"}


def extract_x_value(filename):
    try:
        basename = os.path.basename(filename)
        mask_ratio_str = os.path.splitext(basename)[0].split('_')[-1]
        return float(mask_ratio_str)
    except ValueError:
        return None

def calculate_mean_measure(data):
    total_value = 0
    count = 0
    for task in data.values():
        for language in task.values():
            for value in language.values():
                total_value += value
                count += 1
    return total_value / count if count > 0 else 0

def plot_lineplot_experiments(folder_path, across='tasks', measure="Loss", experiment="Mask"):
    mask_ratios = []
    data_list = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            mask_ratio = extract_x_value(filename)
            if mask_ratio is not None:
                with open(os.path.join(folder_path, filename), 'r') as f:
                    data = json.load(f)
                
                if across == 'tasks':
                    for task, task_data in data.items():
                        mean_measure = calculate_mean_measure({task: task_data})
                        dataset_name = task_codes.get(task)
                        data_list.append({'Category': dataset_name, 'Mean Measure': mean_measure, 'Mask Ratio': mask_ratio})
                        
                elif across == 'languages':
                    for task, lang_data in data.items():
                        for lang, text_data in lang_data.items():
                            full_language_name = language_codes.get(lang)
                            mean_loss = calculate_mean_measure({"temp_task": {lang: text_data}})
                            data_list.append({'Category': full_language_name, 'Mean Measure': mean_loss, 'Mask Ratio': mask_ratio})

                elif across == 'scripts':
                    for task, lang_data in data.items():
                        for lang, text_data in lang_data.items():
                            script_name = language_scripts.get(lang)
                            mean_loss = calculate_mean_measure({"temp_task": {lang: text_data}})
                            data_list.append({'Category': script_name, 'Mean Measure': mean_loss, 'Mask Ratio': mask_ratio})

    df = pd.DataFrame(data_list)

    # For languages, calculate the mean for English categories
    if across == 'languages':
        english_df = df[df['Category'] == 'English']
        english_mean = english_df.groupby('Mask Ratio')['Mean Measure'].mean().reset_index()
        english_mean['Category'] = 'English (Mean)'

        df = df[df['Category'] != 'English']
        df = pd.concat([df, english_mean])

    unique_categories = df['Category'].unique()

    plt.figure(figsize=(10, 6))
    
    if across == 'tasks':
        legend_title = "Dataset"
    elif across == 'languages':
        legend_title = "Language"
    elif across == 'scripts':
        legend_title = "Script"

    ax = sns.lineplot(data=df, x='Mask Ratio', y='Mean Measure', hue='Category',  marker='o', hue_order=unique_categories, linewidth=2, errorbar=None)

    lines = ax.get_lines()
    labels = df['Category'].unique()

    for line, label in zip(lines, labels):
        x_data = line.get_xdata()
        y_data = line.get_ydata()
        x_pos = x_data[0]
        y_pos = y_data[0]
        ax.text(x_pos, y_pos, f' {label}', color=line.get_color(), verticalalignment='baseline', horizontalalignment='center', fontsize=10)


    plt.xlabel('Mask Ratio' if experiment=="Mask" else "Span Length", fontsize=16)
    plt.ylabel(f"Mean {measure}", fontsize=16)
    plt.legend(title=legend_title, fontsize=12, title_fontsize=16, loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.title(what.replace("_", " ").title(), fontsize=18)
    # plt.grid()

    img_name = f"{experiment}_{measure}_{across}_line_plot.pdf"
    plt.savefig(img_name, bbox_inches='tight')
    plt.close()


path_to_folder = "scripts/monte_carlo/results/mask_experiment_1000/loss_scores"
# plot_lineplot_experiments(path_to_folder, across='tasks', measure="Loss", experiment="Mask")
# plot_lineplot_experiments(path_to_folder, across='languages', measure="Loss", experiment="Mask")
plot_lineplot_experiments(path_to_folder, across='scripts', measure="Loss", experiment="Mask")

path_to_folder = "scripts/monte_carlo/results/mask_experiment_1000/std_scores"
# plot_lineplot_experiments(path_to_folder, across='tasks', measure="Uncertainty", experiment="Mask")
# plot_lineplot_experiments(path_to_folder, across='languages', measure="Uncertainty", experiment="Mask")
plot_lineplot_experiments(path_to_folder, across='scripts', measure="Uncertainty", experiment="Mask")

# path_to_folder = "scripts/monte_carlo/results/mask_experiment_1000/gnl_scores"
# plot_lineplot_experiments(path_to_folder, across='tasks', measure="GNLL Loss", experiment="Mask")
# plot_lineplot_experiments(path_to_folder, across='languages', measure="GNLL Loss", experiment="Mask")
# plot_lineplot_experiments(path_to_folder, across='scripts', measure="GNLL Loss", experiment="Mask")

# path_to_folder = "scripts/monte_carlo/results/span_experiment_1000/loss_scores"
# plot_lineplot_experiments(path_to_folder, across='tasks', measure="Loss", experiment="Span")
# plot_lineplot_experiments(path_to_folder, across='languages', measure="Loss", experiment="Span")
# plot_lineplot_experiments(path_to_folder, across='scripts', measure="Loss", experiment="Span")

# path_to_folder = "scripts/monte_carlo/results/span_experiment_1000/std_scores"
# plot_lineplot_experiments(path_to_folder, across='tasks', measure="Uncertainty", experiment="Span")
# plot_lineplot_experiments(path_to_folder, across='languages', measure="Uncertainty", experiment="Span")
# plot_lineplot_experiments(path_to_folder, across='scripts', measure="Uncertainty", experiment="Span")

# path_to_folder = "scripts/monte_carlo/results/span_experiment_1000/gnl_scores"
# plot_lineplot_experiments(path_to_folder, across='tasks', measure="GNLL Loss", experiment="Span")
# plot_lineplot_experiments(path_to_folder, across='languages', measure="GNLL Loss", experiment="Span")
# plot_lineplot_experiments(path_to_folder, across='scripts', measure="GNLL Loss", experiment="Span")