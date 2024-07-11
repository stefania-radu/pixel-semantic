import re
import json
import os

def parse_output_file(file_path):
    task_pattern = re.compile(r"Computing SDs for task: (\w+)")
    language_pattern = re.compile(r"Language: (\w+)")
    data_pattern = re.compile(r"ID text: ([\w-]+) - STD: ([\d.]+) - GNL: ([\d.]+) - Loss: ([\d.]+)")

    results = {}

    with open(file_path, 'r') as file:
        current_task = None
        current_language = None

        for line in file:
            task_match = task_pattern.search(line)
            if task_match:
                current_task = task_match.group(1)
                results[current_task] = {}
                continue

            language_match = language_pattern.search(line)
            if language_match:
                current_language = language_match.group(1)
                if current_language in ['mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'stsb', 'wnli']:
                    continue
                results[current_task][current_language] = {}
                continue

            data_match = data_pattern.search(line)
            if data_match:
                id_text = data_match.group(1)
                std_value = float(data_match.group(2))
                gnl_value = float(data_match.group(3))
                loss_value = float(data_match.group(4))
                
                if current_task and current_language:
                    results[current_task][current_language][id_text] = {
                        "std": std_value,
                        "gnl": gnl_value,
                        "loss": loss_value
                    }

    return results

def save_json(results, metric, output_folder, mask, experiment):
    os.makedirs(output_folder, exist_ok=True)
    output_file_path = os.path.join(output_folder, f'{metric}_per_task_{experiment}_{mask}.json')

    metric_results = {task: {lang: {id_text: data[metric] for id_text, data in lang_data.items()} 
                             for lang, lang_data in task_data.items()} 
                      for task, task_data in results.items()}

    if os.path.exists(output_file_path):
        with open(output_file_path, 'r') as json_file:
            existing_data = json.load(json_file)
            for task, langs in metric_results.items():
                if task in existing_data:
                    for lang, ids in langs.items():
                        if lang in existing_data[task]:
                            existing_data[task][lang].update(ids)
                        else:
                            existing_data[task][lang] = ids
                else:
                    existing_data[task] = langs
            metric_results = existing_data

    with open(output_file_path, 'w') as json_file:
        json.dump(metric_results, json_file, indent=4)
    print(f"File saved: {output_file_path}")


def main():

    experiment = "mask" # CHANGE HERE

    # script was not run for mask 0.1, 0.4 and 0.5
    list_values = [0.1] if experiment == "mask" else [4]

    if experiment == "base":
        list_values = [0.25]

    for mask in list_values:
        mask = str(mask)
        input_file = f"scripts/monte_carlo/results/{experiment}_experiment_1000/std_outputs_{experiment}_cola_{mask}.out"
        results = parse_output_file(input_file)

        # Save JSON files for each metric
        save_json(results, 'std', f'scripts/monte_carlo/results/{experiment}_experiment_1000/std_scores', mask, experiment)
        save_json(results, 'loss', f'scripts/monte_carlo/results/{experiment}_experiment_1000/loss_scores', mask, experiment)
        save_json(results, 'gnl', f'scripts/monte_carlo/results/{experiment}_experiment_1000/gnl_scores', mask, experiment)

if __name__ == "__main__":
    main()
