import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

sns.set_style("darkgrid")


def get_data(path_f1_scores, path_probability_scores):
    data_list = []

    with open(path_f1_scores, 'r') as f:
        f1_data = json.load(f)

    with open(path_probability_scores, 'r') as f:
        confidence_data = json.load(f)

    for lang_id, f1_score in f1_data.items():
        lang = lang_id.split("-")[0].capitalize()
        confidence = confidence_data[lang_id]['probability']
        data_list.append({'Language': lang, 'F1 score': f1_score, 'Confidence': confidence})
 
    df = pd.DataFrame(data_list)

    return df


def plot_hex_calibration(df):
    """Plot measure distributions for each mask ratio."""

    plt.figure(figsize=(10, 8))

    sns.jointplot(df, x='Confidence', y='F1 score', kind="hex", color="#4CB391", height=7)
    plt.plot(np.linspace(0, df['Confidence'].max(), 10), np.linspace(0, df['F1 score'].max(), 10), 'r--')

    plt.xlabel('Confidence', fontsize=18)
    plt.ylabel('F1 score', fontsize=18)
    
    img_name = f"scripts/ensemble/question_answering/results/calibration_plot_qa_hex.png"
    plt.savefig(img_name, bbox_inches='tight')
    plt.close()


def plot_violin(df):

    plt.figure(figsize=(10, 6))

    sns.violinplot(df, x='Language', y='Confidence', split=False, inner='quart', linewidth=1.5, color="#4CB391", cut=0)

    plt.xlabel('Language', fontsize=18)
    plt.ylabel('Confidence', fontsize=18)
    plt.title("Validation Uncertainty in TyDiQA-GoldP", fontsize=20)
    
    img_name = f"scripts/ensemble/question_answering/results/violin_plot_qa.pdf"
    plt.savefig(img_name, bbox_inches='tight')
    plt.close()


def scatter_calibration():
    
    # F1 scores for languages
    f1_scores = {
        "Arabic": 59.46723374105088 / 100,
        "Bengali": 35.12105888212083 / 100,
        "Finnish": 59.5700025964753 / 100,
        "Indonesian": 67.2754986385669 / 100,
        "Swahili": 67.11172284687737 / 100,
        "Korean": 27.141933705371933 / 100,
        "Russian": 53.27744288273116 / 100,
        "Telugu": 63.36311767992068 / 100,
        "English": 62.052267635738836 / 100
    }

    # Confidence scores
    confidence = {
        "Arabic": 0.28510925380254376,
        "Russian": 0.3402623398211962,
        "Bengali": 0.2472791106447368,
        "Telugu": 0.5001047012431127,
        "Finnish": 0.3651030489723315,
        "Swahili": 0.4466881726401327,
        "Korean": 0.19717058117811895,
        "Indonesian": 0.4277023613436089,
        "English": 0.4212115098867394
    }


    data = {
        "Language": list(f1_scores.keys()),
        "F1 Score": list(f1_scores.values()),
        "Confidence": [confidence[lang] for lang in f1_scores.keys()]
    }
    df = pd.DataFrame(data)

    plt.figure(figsize=(10, 6))
    ax = sns.scatterplot(data=df, x="Confidence", y="F1 Score", color="#4CB391", marker="o", s=110)

    adjustments = {
        "Indonesian": {"dx": 0.02, "dy": 0.02},
        "Swahili": {"dx": -0.08, "dy": -0.02}
    }

    for i, row in df.iterrows():
        dx = adjustments.get(row["Language"], {}).get("dx", 0.01)
        dy = adjustments.get(row["Language"], {}).get("dy", 0)
        
        ax.text(row["Confidence"] + dx, row["F1 Score"] + dy, 
                row["Language"], horizontalalignment='left', 
                size='medium', color='black', weight='semibold')

    plt.plot([0, 1], [0, 1], color="red", linestyle="--", label="Perfect Calibration")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Confidence", fontsize=18)
    plt.ylabel("F1 Score", fontsize=18)
    
    img_name = f"scripts/ensemble/question_answering/results/scatter_plot_qa.png"
    plt.savefig(img_name, bbox_inches='tight')
    plt.close()


path_f1_scores = "scripts/ensemble/question_answering/results/f1_scores/f1_scores.json"
path_probability_scores = "scripts/ensemble/question_answering/results/eval_common_predictions_best.json"

df = get_data(path_f1_scores, path_probability_scores)

plot_violin(df)
# scatter_calibration()
# plot_hex_calibration(df)
