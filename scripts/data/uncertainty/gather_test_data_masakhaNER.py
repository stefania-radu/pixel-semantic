# import os
# import json

# # Base path to MasakhaNER2.0 data
# base_path = 'data/masakhane-ner/data'

# # Initialize an empty dictionary
# elements_dict = {}

# # Function to process each test.txt file
# def process_file(file_path, language_key):
#     with open(file_path, 'r', encoding='utf-8') as file:
#         content = file.read().strip()
#         elements = content.split('\n\n')  # Split by empty line between elements
#         for idx, element in enumerate(elements):
#             key = f"{language_key}_{idx}"
#             lines = element.split('\n')
#             cleaned_lines = [line.split()[0] for line in lines if len(line.split()) == 2]  # Exclude labels
#             elements_dict[key] = '\n'.join(cleaned_lines)

# # Iterate through directories within the base path
# for language_dir in os.listdir(base_path):
#     language_path = os.path.join(base_path, language_dir)
#     if os.path.isdir(language_path):
#         test_file_path = os.path.join(language_path, 'test.txt')
#         if os.path.exists(test_file_path):
#             process_file(test_file_path, language_dir)

# # Save the dictionary to a JSON file
# with open('masakhaner_test_data_for_rendering.json', 'w', encoding='utf-8') as json_file:
#     json.dump(elements_dict, json_file, ensure_ascii=False, indent=2)

# print(f"Dictionary created with {len(elements_dict)} elements.")



############## Save only N examples per language ################
# python process_masakhane.py 10

import os
import json
import argparse
import random

# Parse command line arguments
parser = argparse.ArgumentParser(description='Process MasakhaNER data.')
parser.add_argument('N', type=int, help='Number of random examples per language')
args = parser.parse_args()

N = args.N  # Number of examples to select

# Base path to MasakhaNER2.0 data
base_path = 'data/masakhane-ner/data'

# Initialize an empty dictionary
elements_dict = {}

# Function to process each test.txt file
def process_file(file_path, language_key, N):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read().strip()
        elements = content.split('\n\n')  # Split by empty line between elements
        # Select N random examples
        selected_elements = random.sample(elements, min(N, len(elements)))
        for idx, element in enumerate(selected_elements):
            key = f"{language_key}_{idx}"
            lines = element.split('\n')
            cleaned_lines = [line.split()[0] for line in lines if len(line.split()) == 2]  # Exclude labels
            elements_dict[key] = '\n'.join(cleaned_lines)

# Iterate through directories within the base path
for language_dir in os.listdir(base_path):
    language_path = os.path.join(base_path, language_dir)
    if os.path.isdir(language_path):
        test_file_path = os.path.join(language_path, 'test.txt')
        if os.path.exists(test_file_path):
            process_file(test_file_path, language_dir, N)

# Save the dictionary to a JSON file
with open('masakhaner_test_data_for_rendering.json', 'w', encoding='utf-8') as json_file:
    json.dump(elements_dict, json_file, ensure_ascii=False, indent=4)

print(f"Dictionary created with {len(elements_dict)} elements.")

