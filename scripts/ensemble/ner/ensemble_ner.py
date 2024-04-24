import os
import pandas as pd
import numpy as np
import ast


def get_argmax_label(predictions_probs):
    labels = ['O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG', 'B-MISC', 'I-MISC']
    return labels[np.argmax(predictions_probs)]

base_dir = 'NER-results'

for language in os.listdir(base_dir):
    language_dir = os.path.join(base_dir, language)
    if os.path.isdir(language_dir):
        ensemble_data = None

        for model_dir in os.listdir(language_dir):
            model_path = os.path.join(language_dir, model_dir)
            if os.path.isdir(model_path):
                for file in os.listdir(model_path):
                    if file.endswith('.csv'):
                        file_path = os.path.join(model_path, file)
                       
                        data = pd.read_csv(file_path)
                       
                        data['predictions_probs'] = data['predictions_probs'].apply(ast.literal_eval)
                        
                        if ensemble_data is None:
                            ensemble_data = data.copy()
                        else:
                           
                            ensemble_data['predictions_probs'] += data['predictions_probs']

      
        ensemble_data['predictions_probs'] = ensemble_data['predictions_probs'].apply(lambda x: np.array(x) / len(os.listdir(language_dir)))

        
        ensemble_data['predictions_probs'] = ensemble_data['predictions_probs'].apply(list)

        
        ensemble_data.to_csv(os.path.join(language_dir, 'ensemble_predictions.csv'), index=False)

        print(f"Ensemble predictions for {language} saved to {os.path.join(language_dir, 'ensemble_predictions.csv')}")
