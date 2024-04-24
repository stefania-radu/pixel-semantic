############## Save only N examples per language ################
# python scripts\data\uncertainty\gather_test_data_masakhaNER.py 1000 test_data_for_rendering_masakhaner_1000.json

import os
import json
import argparse
import random

random.seed(42)

parser = argparse.ArgumentParser(description='Process MasakhaNER data.')
parser.add_argument('N', type=int, help='Number of random examples per language')
parser.add_argument('file_name', type=str, default="test_data_for_rendering_masakhaner.json", help='name of file to be saved')
args = parser.parse_args()

N = args.N 
file_name = args.file_name

base_path = '/scratch/s3919609/data/masakhane-ner/data'

elements_dict = {}

def process_file(file_path, language_key, N):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read().strip()
        elements = content.split('\n\n')
        # Select N random examples
        selected_elements = random.sample(elements, min(N, len(elements)))
        elements_dict[language_key] = {} 
        for idx, element in enumerate(selected_elements):
            key = f"{language_key}_{idx}" 
            lines = element.split('\n')
            cleaned_lines = [line.split()[0] for line in lines if len(line.split()) == 2]  # Exclude labels
            elements_dict[language_key][key] = '\n'.join(cleaned_lines)  # Nested dict entry

for language_dir in os.listdir(base_path):
    language_path = os.path.join(base_path, language_dir)
    if os.path.isdir(language_path):
        test_file_path = os.path.join(language_path, 'test.txt')
        if os.path.exists(test_file_path):
            process_file(test_file_path, language_dir, N)

element_counts = {key: len(value) for key, value in elements_dict.items()}
print(f"Counts: {element_counts}") # Counts: {'amh': 500, 'conll_2003_en': 1000, 'hau': 552, 'ibo': 638, 'kin': 605, 'lug': 407, 'luo': 186, 'pcm': 600, 'swa': 604, 'wol': 539, 'yor': 645, 'zh': 1000}

with open(file_name, 'w', encoding='utf-8') as json_file:
    json.dump(elements_dict, json_file, ensure_ascii=False, indent=4)

print(f"Data saved to {file_name}")
