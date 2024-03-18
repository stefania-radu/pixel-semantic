####### Combine small test sets for all 3 tasks into 1###########
# python scripts\data\uncertainty\combine_test_data.py

import json

# Define the filenames for the input dictionaries
ner_filename = 'test_data_for_rendering_masakhaner.json'
tydiqa_filename = 'test_data_for_rendering_tydiqa.json'
glue_filename = 'test_data_for_rendering_glue.json'

# Load the dictionaries
with open(ner_filename, 'r', encoding='utf-8') as f:
    ner_data = json.load(f)

with open(tydiqa_filename, 'r', encoding='utf-8') as f:
    tydiqa_data = json.load(f)

with open(glue_filename, 'r', encoding='utf-8') as f:
    glue_data = json.load(f)

# Combine the dictionaries under their task names
combined_data = {
    "ner": ner_data,
    "tydiqa": tydiqa_data,
    "glue": glue_data
}

# Save the combined dictionary to a JSON file
combined_filename = 'test_data_ner_tydiqa_glue_small.json'
with open(combined_filename, 'w', encoding='utf-8') as f:
    json.dump(combined_data, f, ensure_ascii=False, indent=4)

print(f"Combined data saved to {combined_filename}.")
