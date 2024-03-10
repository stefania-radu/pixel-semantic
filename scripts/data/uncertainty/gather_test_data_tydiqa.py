import json
from datasets import load_dataset

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

# Initialize an empty dictionary
concatenated_texts = {}

# Iterate through the validation dataset
for example in raw_datasets["validation"]:
    # Concatenate question and context
    concatenated_text = example[question_column_name] + '\n' + example[context_column_name]
    # Use the id as the key and the concatenated text as the value
    concatenated_texts[example['id']] = concatenated_text

# The concatenated_texts dictionary now contains the desired mapping
print(f"Number of entries in the dictionary: {len(concatenated_texts)}")

# Optionally, to display a few entries
for idx, (key, value) in enumerate(concatenated_texts.items()):
    print(f"ID: {key}, Text: {value[:100]}")  # Displaying only the first 100 characters for brevity
    if idx == 5:  # Limit to showing first few entries
        break

# File name according to the specified format
file_name = f"{dataset_name}_test_data_for_rendering.json"

# Saving the dictionary to a JSON file
with open(file_name, 'w', encoding='utf-8') as f:
    json.dump(concatenated_texts, f, ensure_ascii=False, indent=4)

print(f"Dictionary saved to {file_name}")