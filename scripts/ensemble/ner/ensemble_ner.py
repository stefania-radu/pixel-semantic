import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import softmax
from sklearn.metrics import f1_score
from scipy.stats import gaussian_kde

sns.set_style("darkgrid")


language_dict = {
    'amh': 'Amharic',
    'hau': 'Hausa',
    'ibo': 'Igbo',
    'kin': 'Kinyarwanda',
    'lug': 'Ganda',
    'luo': 'Luo',
    'pcm': 'Nigerian Pidgin',
    'swa': 'Swahili',
    'wol': 'Wolof',
    'yor': 'Yoruba'
}

def read_csv_files(folder_path):
    """
    Reads all .csv files from the specified folder and returns a dictionary of dataframes.
    Each dataframe corresponds to a language, with columns: word, label, pred, logits.
    """
    dataframes = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            # Extract language from file name
            language = file_name.split('_')[-1].replace('.csv', '')
            df = pd.read_csv(os.path.join(folder_path, file_name))
            df.columns = ['word', 'label', 'pred', 'logits']
            dataframes[language] = df

    print("Done reading")
    return dataframes


def add_prediction_probs(dataframes):

    for language, df in dataframes.items():
        logits = np.array(df['logits'].apply(eval).tolist())
        probs = softmax(logits, axis=1)
        df['prediction_probs'] = list(probs)
        df['confidence'] = df['prediction_probs'].apply(max)
        # exit()
        df.to_csv(f"NER-results/ensemble_predictions_test_softmax_{language}.csv", index=False)

    return dataframes


def compute_f1_scores(dataframes):

    f1_scores = {}
    for language, df in dataframes.items():
        true_labels = df['label']
        predictions = df['pred']
        f1_scores[language_dict[language]] = f1_score(true_labels, predictions, average='weighted')

    f1_scores['avg'] = np.mean(list(f1_scores.values()))
    print(f1_scores)
    return f1_scores


def plot_violin(dataframes):
    label_order = ['O', 'B-ORG', 'I-ORG', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-DATE', 'I-DATE']
    color_palette = sns.color_palette("tab10", len(label_order))
    label_color_mapping = {label: color_palette[i] for i, label in enumerate(label_order)}

    for language, df in dataframes.items():
        print(language_dict[language])
        
        class_counts = df['label'].value_counts().to_dict()
        print(class_counts)

        # Create a mapping of original labels to labels with counts
        label_mapping = {label: f"{label}: {count}" for label, count in class_counts.items()}
        
        # Map the labels in the DataFrame
        df['Label Counts'] = df['label'].map(label_mapping)

        unique_labels = df['Label Counts'].unique()
        unique_palette = {label: label_color_mapping[label.split(':')[0]] for label in unique_labels}

        # Create a violin plot with updated labels
        plt.figure(figsize=(10, 6))
        ax = sns.violinplot(
            x='label', y='confidence', hue='Label Counts', data=df, 
            density_norm='width', split=False, inner='quart', cut=0, 
            order=label_order, palette=unique_palette
        )

        ax.set_ylim(0, 1.0)

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        sorted_handles = [by_label[f"{label}: {class_counts[label]}"] for label in label_order if f"{label}: {class_counts[label]}" in by_label]
        sorted_labels = [f"{label}: {class_counts[label]}" for label in label_order if f"{label}: {class_counts[label]}" in by_label]
        ax.legend(sorted_handles, sorted_labels, loc='lower left')

        plt.xlabel('Label', fontsize=18)
        plt.ylabel('Confidence', fontsize=18)
        plt.title(f"Language: {language_dict[language]}", fontsize=20)
        
        img_name = f"scripts/ensemble/ner/violin_plot_ner_{language}.pdf"
        plt.savefig(img_name, bbox_inches='tight')
        plt.close()


def plot_confidences_kde(dataframes):
    def weighted_kde(data, weights, grid):
        data = np.asarray(data, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        kde = gaussian_kde(data, weights=weights)
        return kde(grid)

    grid = np.linspace(0, 1, 1000)
    
    for language, df in dataframes.items():
        print(f"Language: {language}")

        # Compute class counts and weights
        class_counts = df['label'].value_counts()
        df['weight'] = df['label'].apply(lambda x: 1 / class_counts[x])

        # Create a label to index mapping based on the unique labels in the dataset
        unique_labels = sorted(df['label'].unique())
        label_to_index = {label: idx for idx, label in enumerate(unique_labels)}

        # Extract the relevant confidence value for each prediction
        def get_confidence(row):
            pred_index = label_to_index[row['pred']]
            return row['prediction_probs'][pred_index]

        df['confidence'] = df.apply(get_confidence, axis=1)

        print(f"Confidences for {language}:")
        print(df['confidence'].describe())
        
        # Compute weighted KDE for each class
        kde_data = []
        for label, group in df.groupby('label'):
            weights = list(group['weight'])
            confidences = list(group['confidence'])
            kde = weighted_kde(confidences, weights, grid)
            kde_data.append(pd.DataFrame({
                'language': language,
                'label': label,
                'confidence': grid,
                'density': kde
            }))
        
        # Combine KDE data for the current language
        kde_df = pd.concat(kde_data).reset_index(drop=True)
        
        # Plot KDE for the current language
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=kde_df, x='confidence', y='density', hue='label')
        plt.title(f'Confidence Distribution by Class for {language}')
        plt.xlabel('Confidence')
        plt.ylabel('Density')
        plt.title("Validation Uncertainty in NER", fontsize=20)
        
        img_name = f"scripts/ensemble/ner/kde_plot_ner_{language}.pdf"
        plt.savefig(img_name, bbox_inches='tight')
        plt.close()

folder_path = 'NER-results'
dataframes = read_csv_files(folder_path)
dataframes = add_prediction_probs(dataframes)
compute_f1_scores(dataframes)
# plot_confidences_kde(dataframes)
plot_violin(dataframes)
