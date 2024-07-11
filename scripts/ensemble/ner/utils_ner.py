import os
import pandas as pd
import numpy as np
import ast
import re

def get_argmax_label(predictions_probs):
    labels = ['O', 'B-ORG', 'I-ORG', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-DATE', 'I-DATE']
    return labels[np.argmax(predictions_probs)]

def preprocess_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    new_lines = []
    temp_line = ""
    for line in lines:
        if line.strip() and not line.startswith(' '): 
            if temp_line:
                new_lines.append(temp_line.strip() + '\n')
            temp_line = line.strip()
        elif line.startswith(' '):
            temp_line += ' ' + ' '.join(line.split())

    if temp_line:
        new_lines.append(temp_line)

    return new_lines

def load_and_process_data(file_path):
    clean_lines = preprocess_file(file_path)

    temp_file_path = file_path + '.temp'
    with open(temp_file_path, 'w', encoding='utf-8') as temp_file:
        temp_file.writelines(clean_lines)

    data = pd.read_csv(temp_file_path, sep='\t', engine='python')
    # data['predictions_probs'] = data['predictions_probs'].apply(safely_eval)
    data['predictions_probs'] = data['predictions_probs'].apply(string_to_list)
    os.remove(temp_file_path)
    return data


def string_to_list(s):
    s = s.strip('[]')
    
    num_list = [float(num) for num in s.split()]
    
    return num_list

base_dir = '/scratch/s3919609/Pixel-stuff/Pixel-stuff/results/NER-results'

for language in os.listdir(base_dir):
    language_dir = os.path.join(base_dir, language)
    if os.path.isdir(language_dir):
        ensemble_data = None

        for model_dir in os.listdir(language_dir):
            print(model_dir)
            model_path = os.path.join(language_dir, model_dir)
            if os.path.isdir(model_path):
                for file in os.listdir(model_path):
                    if file.endswith('test_predictions.csv'):
                        file_path = os.path.join(model_path, file)

                        data = load_and_process_data(file_path)

                        if ensemble_data is None:
                            ensemble_data = data.copy()
                        else:
                            # summing arrays in the 'predictions_probs' columns
                            ensemble_data['predictions_probs'] = [
                                np.array(a) + np.array(b)
                                for a, b in zip(ensemble_data['predictions_probs'], data['predictions_probs'])
                            ]

        if ensemble_data is not None:
            ensemble_data['predictions_probs'] = ensemble_data['predictions_probs'].apply(lambda x: list(np.array(x) / len(os.listdir(language_dir))))

            ensemble_output_path = os.path.join(language_dir, f'ensemble_predictions_test.csv')
            ensemble_data.to_csv(ensemble_output_path, index=False)
            print(f"Ensemble predictions for {language} saved to {ensemble_output_path}")
