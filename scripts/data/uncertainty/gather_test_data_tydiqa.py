########### Save only N random examples #############
# python scripts/data/uncertainty/gather_test_data_tydiqa.py 1000 test_data_for_rendering_tydiqa_1000.json

import json
import argparse
from datasets import load_dataset
import random

random.seed(42)

parser = argparse.ArgumentParser(description='Process TyDi QA data.')
parser.add_argument('N', type=int, help='Number of random examples per language')
parser.add_argument('file_name', type=str, default="test_data_for_rendering_tydiqa_1000.json", help='name of file to be saved')
args = parser.parse_args()

N = args.N
file_name = args.file_name

dataset_name = "tydiqa"
dataset_config_name = "secondary_task"

raw_datasets = load_dataset(
            dataset_name,
            dataset_config_name,
            trust_remote_code=True
        )

column_names = raw_datasets["validation"].column_names

question_column_name = "question" if "question" in column_names else column_names[0]
context_column_name = "context" if "context" in column_names else column_names[1]

concatenated_texts_by_language = {}

for example in raw_datasets["validation"]:
    # Extract language from the example ID
    language = example['id'].split("-")[0]
    # Concatenate question and context
    concatenated_text = example[question_column_name] + '\n' + example[context_column_name]
    # Organize concatenated texts by language
    if language not in concatenated_texts_by_language:
        concatenated_texts_by_language[language] = []
    concatenated_texts_by_language[language].append((example['id'], concatenated_text))

final_selection = {}


for language, texts in concatenated_texts_by_language.items():
    selected_texts = random.sample(texts, min(len(texts), N))
    final_selection[language] = {}
    for id, text in selected_texts:
        final_selection[language][id] = text

element_counts = {key: len(value) for key, value in final_selection.items()}
print(f"Counts: {element_counts}") # Counts: {'arabic': 921, 'russian': 812, 'bengali': 113, 'telugu': 669, 'finnish': 782, 'swahili': 499, 'korean': 276, 'indonesian': 565, 'english': 440}

with open(file_name, 'w', encoding='utf-8') as f:
    json.dump(final_selection, f, ensure_ascii=False, indent=4)

print(f"Data saved to {file_name}.")

