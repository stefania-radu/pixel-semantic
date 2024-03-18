# import json
# from datasets import load_dataset

# # Task to keys mapping
# task_to_keys = {
#     "cola": ("sentence", None),
#     "mnli": ("premise", "hypothesis"),
#     "mrpc": ("sentence1", "sentence2"),
#     "qnli": ("question", "sentence"),
#     "qqp": ("question1", "question2"),
#     "rte": ("sentence1", "sentence2"),
#     "sst2": ("sentence", None),
#     "stsb": ("sentence1", "sentence2"),
#     "wnli": ("sentence1", "sentence2"),
# }

# # Initialize a dictionary for the concatenated strings
# concatenated_strings = {}

# # Load dataset for each task and process accordingly
# for task, keys in task_to_keys.items():
#     # Use the test_matched split for mnli and test for other tasks
#     split = "test_matched" if task == "mnli" else "test"
#     try:
#         dataset = load_dataset("glue", task, split=split)
#     except Exception as e:
#         print(f"Could not load dataset for task {task} with split {split}: {e}")
#         continue
    
#     for idx, item in enumerate(dataset):
#         # Generate an ID with sequential numbers
#         item_id = f"{task}_en_{idx}"
#         # Concatenate strings based on specified columns
#         concatenated_string = item[keys[0]] if keys[0] in item else ""
#         if keys[1] and keys[1] in item:
#             concatenated_string += f"\n{item[keys[1]]}" if concatenated_string else item[keys[1]]
#         # Add to the dictionary
#         concatenated_strings[item_id] = concatenated_string

# # Save the dictionary as a JSON file
# with open('test_data_for_rendering_glue.json', 'w', encoding='utf-8') as f:
#     json.dump(concatenated_strings, f, ensure_ascii=False, indent=4)

# print(f"Data saved to test_data_for_rendering_glue.json")


########## Save N examples for each GLUE task #############
# python process_glue.py 10

import json
import argparse
from datasets import load_dataset
import random

parser = argparse.ArgumentParser(description='Process GLUE tasks data.')
parser.add_argument('N', type=int, help='Number of random examples per task')
args = parser.parse_args()

N = args.N  # Number of examples to select per task

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

concatenated_strings = {}

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
        item_id = f"{task}_en_{idx}"
        concatenated_string = item[keys[0]] if keys[0] in item else ""
        if keys[1] and keys[1] in item:
            concatenated_string += f"\n{item[keys[1]]}" if concatenated_string else item[keys[1]]
        concatenated_strings[item_id] = concatenated_string

with open('test_data_for_rendering_glue.json', 'w', encoding='utf-8') as f:
    json.dump(concatenated_strings, f, ensure_ascii=False, indent=4)

print(f"Data saved to test_data_for_rendering_glue.json with {len(concatenated_strings)} entries.")
