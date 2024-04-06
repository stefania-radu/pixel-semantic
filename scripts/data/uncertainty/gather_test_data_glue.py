########## Save N examples for each GLUE task #############
# python scripts/data/uncertainty/gather_test_data_glue.py 1000 test_data_for_rendering_glue_1000.json

import json
import argparse
from datasets import load_dataset
import random

random.seed(42)

parser = argparse.ArgumentParser(description='Process GLUE tasks data.')
parser.add_argument('N', type=int, help='Number of random examples per task')
parser.add_argument('file_name', type=str, default="test_data_for_rendering_glue_1000.json", help='name of file to be saved')
args = parser.parse_args()

N = args.N
file_name = args.file_name

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

# Initialize concatenated_strings with a nested structure
concatenated_strings = {task: {} for task in task_to_keys}

for task, keys in task_to_keys.items():
    # Use the test_matched split for mnli and test for other tasks
    split = "test_matched" if task == "mnli" else "test"
    try:
        dataset = load_dataset("glue", task, split=split)
    except Exception as e:
        print(f"Could not load dataset for task {task} with split {split}: {e}")
        continue
    
    selected_indices = random.sample(range(len(dataset)), min(N, len(dataset)))
    
    for idx in selected_indices:
        item = dataset[idx]
        item_id = f"{task}_{idx}"  # Use idx as the key for simplicity
        concatenated_string = item[keys[0]] if keys[0] in item else ""
        if keys[1] and keys[1] in item:
            concatenated_string += f"\n{item[keys[1]]}" if concatenated_string else item[keys[1]]
        concatenated_strings[task][item_id] = concatenated_string  # Place under task-specific dictionary

with open(file_name, 'w', encoding='utf-8') as f:
    json.dump(concatenated_strings, f, ensure_ascii=False, indent=4)

element_counts = {key: len(value) for key, value in concatenated_strings.items()}
print(f"Counts: {element_counts}") # Counts: {'cola': 1000, 'mnli': 1000, 'mrpc': 1000, 'qnli': 1000, 'qqp': 1000, 'rte': 1000, 'sst2': 1000, 'stsb': 1000, 'wnli': 146}

print(f"Data saved to {file_name}")

