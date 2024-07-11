import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="darkgrid")

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

def plot_loss_vs_mask_ratio(path_to_loss, path_to_SD, across='tasks'):
    data_list = []

    mask_ratio = extract_x_value(path_to_loss)

    with open(path_to_loss, 'r') as f:
        loss_data = json.load(f)

    with open(path_to_SD, 'r') as f:
        SD_data = json.load(f)
    
    if across == 'tasks':
        for (task_loss, task_data_loss), (task_sd, task_data_sd) in zip(loss_data.items(), SD_data.items()):
            for (lang, text_data_loss), (lang, text_data_sd) in zip(task_data_loss.items(), task_data_sd.items()):
                for (_, value_loss), (_, value_sd) in zip(text_data_loss.items(), text_data_sd.items()):
                    dataset_name = task_codes.get(task_loss)
                    value_loss = value_loss ** 0.5 # get RMSE
                    data_list.append({'Category': dataset_name, 'Loss': value_loss, 'Uncertainty': value_sd, 'Mask Ratio': mask_ratio})
    else:
        for (task_loss, task_data_loss), (task_sd, task_data_sd) in zip(loss_data.items(), SD_data.items()):
            for (lang, text_data_loss), (lang, text_data_sd) in zip(task_data_loss.items(), task_data_sd.items()):
                full_language_name = language_codes.get(lang)
                for (_, value_loss), (_, value_sd) in zip(text_data_loss.items(), text_data_sd.items()):
                    value_loss = value_loss ** 0.5 # get RMSE
                    data_list.append({'Category': full_language_name, 'Loss': value_loss, 'Uncertainty': value_sd, 'Mask Ratio': mask_ratio})

    df = pd.DataFrame(data_list)

    plt.figure(figsize=(10, 8))
    
    if across == 'tasks':
        legend_title = "Dataset"
    else:
        legend_title = "Language"
    # kdeplot sns
    
    sns.kdeplot(data=df, x='Uncertainty', y='Loss', fill=True, thresh=0.01, color="#4CB391")
    sns.scatterplot(data=df, x='Uncertainty', y='Loss', color='red', hue='Category', style='Category')

    plt.xlabel('Uncertainty (SD)', fontsize=18)
    plt.ylabel('Loss (RMSE)', fontsize=18)
    plt.legend(title=legend_title, fontsize=14, title_fontsize=16, loc='center left', bbox_to_anchor=(1, 0.5))

    img_name = f"calibration_plot_{across}_{mask_ratio}_kde.pdf"
    plt.savefig(img_name, bbox_inches='tight')
    plt.clf()


    # hexplot sns
    
    sns.jointplot(df, x='Uncertainty', y='Loss', kind="hex", color="#4CB391", height=7)
    # add line for perfect calibration
    # plt.plot(np.linspace(0.07, df['Uncertainty'].max(), 10), np.linspace(0, df['Loss'].max(), 10), 'r--')
    plt.gcf().axes[0].set_xlim(left=0.07)
    plt.xlabel('Uncertainty (SD)', fontsize=18)
    plt.ylabel('Loss (RMSE)', fontsize=18)
    
    img_name = f"calibration_plot_{mask_ratio}_hex.png"
    plt.savefig(img_name, bbox_inches='tight')
    plt.close()
    

path_to_loss = "scripts/monte_carlo/results/base_experiment_1000/loss_scores/loss_per_task_base_0.25.json"
path_to_SD = "scripts/monte_carlo/results/base_experiment_1000/std_scores/std_per_task_base_0.25.json"

plot_loss_vs_mask_ratio(path_to_loss, path_to_SD, across='tasks')
# plot_loss_vs_mask_ratio(path_to_loss, path_to_SD, across='languages')
