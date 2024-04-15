import argparse
import json
import os
from collections import defaultdict

# python scripts/ensemble/compute_confidence.py scripts/ensemble/results/eval_common_predictions_best.json scripts/ensemble/results/confidence_scores.json

def read_and_aggregate_confidence_scores(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)

    language_probabilities = defaultdict(list)
    for example_id, info in data.items():
        language = example_id.split('-')[0]
        language_probabilities[language].append(info["probability"])

    average_probabilities = {}
    for language, probs in language_probabilities.items():
        avg_prob = sum(probs) / len(probs)
        average_probabilities[language.capitalize()] = avg_prob

    with open(output_file, 'w') as f:
        json.dump(average_probabilities, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute average confidence scores for each language.")
    parser.add_argument("input_file", help="Path to the input JSON file containing predictions.")
    parser.add_argument("output_file", help="Path to the input JSON file containing predictions.")
    
    args = parser.parse_args()
    
    read_and_aggregate_confidence_scores(args.input_file, args.output_file)
