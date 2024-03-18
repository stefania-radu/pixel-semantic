# import json
# from datasets import load_dataset

# dataset_name = "tydiqa"
# dataset_config_name = "secondary_task"

# raw_datasets = load_dataset(
#             dataset_name,
#             dataset_config_name,
#             use_auth_token=None,
#             ignore_verifications=True,
#         )

# column_names = raw_datasets["validation"].column_names

# question_column_name = "question" if "question" in column_names else column_names[0]
# context_column_name = "context" if "context" in column_names else column_names[1]

# # Initialize an empty dictionary
# concatenated_texts = {}

# # Iterate through the validation dataset
# for example in raw_datasets["validation"]:
#     # Concatenate question and context
#     concatenated_text = example[question_column_name] + '\n' + example[context_column_name]
#     # Use the id as the key and the concatenated text as the value
#     concatenated_texts[example['id']] = concatenated_text

# # The concatenated_texts dictionary now contains the desired mapping
# print(f"Number of entries in the dictionary: {len(concatenated_texts)}")

# # Optionally, to display a few entries
# for idx, (key, value) in enumerate(concatenated_texts.items()):
#     print(f"ID: {key}, Text: {value[:100]}")  # Displaying only the first 100 characters for brevity
#     if idx == 5:  # Limit to showing first few entries
#         break

# # File name according to the specified format
# file_name = f"{dataset_name}_test_data_for_rendering.json"

# # Saving the dictionary to a JSON file
# with open(file_name, 'w', encoding='utf-8') as f:
#     json.dump(concatenated_texts, f, ensure_ascii=False, indent=4)

# print(f"Dictionary saved to {file_name}")


########### Save only N random examples #############
# python gather_test_data_tydiqa.py 10

import json
import argparse
from datasets import load_dataset
import random

# Parse command line arguments
parser = argparse.ArgumentParser(description='Process TyDi QA data.')
parser.add_argument('N', type=int, help='Number of random examples per language')
args = parser.parse_args()

N = args.N  # Number of examples to select per language

dataset_name = "tydiqa"
dataset_config_name = "secondary_task"

raw_datasets = load_dataset(
            dataset_name,
            dataset_config_name,
            use_auth_token=None,
            ignore_verifications=True,
        )

column_names = raw_datasets["validation"].column_names

question_column_name = "question" if "question" in column_names else column_names[0]
context_column_name = "context" if "context" in column_names else column_names[1]

# Initialize an empty dictionary to hold concatenated texts organized by language
concatenated_texts_by_language = {}

# Iterate through the validation dataset
for example in raw_datasets["validation"]:
    # Extract language from the example ID
    language = example['id'].split("-")[0]
    # Concatenate question and context
    concatenated_text = example[question_column_name] + '\n' + example[context_column_name]
    # Organize concatenated texts by language
    if language not in concatenated_texts_by_language:
        concatenated_texts_by_language[language] = []
    concatenated_texts_by_language[language].append((example['id'], concatenated_text))

# Initialize an empty dictionary for the final selection
final_selection = {}

# Select N random examples for each language
for language, texts in concatenated_texts_by_language.items():
    selected_texts = random.sample(texts, min(len(texts), N))
    for id, text in selected_texts:
        final_selection[id] = text

# Saving the dictionary to a JSON file
file_name = f"{dataset_name}_test_data_for_rendering.json"
with open(file_name, 'w', encoding='utf-8') as f:
    json.dump(final_selection, f, ensure_ascii=False, indent=4)

print(f"Dictionary saved to {file_name} with {len(final_selection)} entries.")
