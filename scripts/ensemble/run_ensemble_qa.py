import argparse
import os
import json
from collections import defaultdict

# python scripts/ensemble/run_ensemble_qa.py results_small/tydiqa scripts/ensemble/results 2

def read_predictions(folder_path):
    model_predictions = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file == 'eval_nbest_predictions.json':
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    predictions = json.load(f)
                    model_predictions.append(predictions)

    print("Done reading predictions")
    return model_predictions


def find_common_answers(model_predictions, common_cutoff):
    answer_probabilities = defaultdict(list)
    highest_prob_answer_per_example = {}
    
    #track highest probability answer for each example
    for predictions in model_predictions:
        for example_id, answers in predictions.items():
            for answer in answers:
                key = (example_id, answer["text"])
                answer_probabilities[key].append(answer["probability"])
                # Update the highest probability answer if this one is higher
                if example_id not in highest_prob_answer_per_example or highest_prob_answer_per_example[example_id]["probability"] < answer["probability"]:
                    highest_prob_answer_per_example[example_id] = {"text": answer["text"], "probability": answer["probability"]}

    # common answers or fall back to highest probability answer
    common_answers = {}
    for example_id, _ in highest_prob_answer_per_example.items():
        filtered_answers = [key for key, probs in answer_probabilities.items() if key[0] == example_id and len(probs) >= common_cutoff]
        if filtered_answers:
            for key in filtered_answers:
                _, text = key
                avg_prob = sum(answer_probabilities[key]) / len(answer_probabilities[key])
                if example_id not in common_answers:
                    common_answers[example_id] = []
                common_answers[example_id].append({"text": text, "probability": avg_prob})
        else:
            #use the highest probability answer
            common_answers[example_id] = [highest_prob_answer_per_example[example_id]]

    print("Done finding the common answers")
    return common_answers



def save_common_answers(common_answers, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for example_id in common_answers:
        common_answers[example_id] = sorted(common_answers[example_id], key=lambda x: x["probability"], reverse=True)

    output_file = os.path.join(output_folder, 'eval_common_predictions.json')
    with open(output_file, 'w') as f:
        json.dump(common_answers, f, ensure_ascii=False, indent=4)
        
    print(f"Saved in {output_file}")


def save_best_common_answers(common_answers, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    best_common_answers = {}
    for example_id, answers in common_answers.items():
        best_common_answers[example_id] = answers[0]

    output_file = os.path.join(output_folder, 'eval_common_predictions_best.json')
    with open(output_file, 'w') as f:
        json.dump(best_common_answers, f, ensure_ascii=False, indent=4)


def main(folder_path, output_folder, common_cutoff):
    model_predictions = read_predictions(folder_path)
    common_answers = find_common_answers(model_predictions, common_cutoff)
    save_common_answers(common_answers, output_folder)
    save_best_common_answers(common_answers, args.output_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find common answers from TyDi QA dataset predictions across multiple models.")
    parser.add_argument("input_folder", help="Path to the input folder containing model prediction folders.")
    parser.add_argument("output_folder", help="Path to the folder where the common predictions JSON will be saved.")
    parser.add_argument("common_cutoff", type=int, help="The minimum number of models that must agree on an answer for it to be considered 'common'.")
    
    args = parser.parse_args()
    
    main(args.input_folder, args.output_folder, args.common_cutoff)
