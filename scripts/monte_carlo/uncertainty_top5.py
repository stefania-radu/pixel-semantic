import json

def find_extreme_loss_ids(value=5):
    file_path = 'scripts/monte_carlo/results/base_experiment_1000/loss_scores/loss_per_task_base_0.25.json'


    with open(file_path, 'r') as file:
        losses_per_task = json.load(file)
    
    losses_dict = {}
    overall_lowest_losses = []
    overall_highest_losses = []

    for task, langs in losses_per_task.items():
        for lang, id_losses in langs.items():

            sorted_losses = sorted(id_losses.items(), key=lambda x: x[1])

            lowest_losses = sorted_losses[:value]
            highest_losses = sorted_losses[-value:]

            overall_lowest_losses.extend(lowest_losses)
            overall_highest_losses.extend(highest_losses)

            if lang not in losses_dict:
                losses_dict[lang] = {}
            losses_dict[lang]['low'] = {id_text: loss for id_text, loss in lowest_losses}
            losses_dict[lang]['high'] = {id_text: loss for id_text, loss in highest_losses}

    overall_lowest_losses = sorted(overall_lowest_losses, key=lambda x: x[1])[:value]
    overall_highest_losses = sorted(overall_highest_losses, key=lambda x: x[1], reverse=True)[:value]

    for lang in losses_dict:
        print(f"Language: {lang}")
        print("Lowest losses:")
        for id_text, loss in losses_dict[lang]['low'].items():
            print(f"ID: {id_text}, Loss: {loss}")
        print("Highest losses:")
        for id_text, loss in losses_dict[lang]['high'].items():
            print(f"ID: {id_text}, Loss: {loss}")

    print("\nOverall lowest losses:")
    for id_text, loss in overall_lowest_losses:
        print(f"ID: {id_text}, Loss: {loss}")

    print("Overall highest losses:")
    for id_text, loss in overall_highest_losses:
        print(f"ID: {id_text}, Loss: {loss}")

    return losses_dict, {'overall_low': overall_lowest_losses, 'overall_high': overall_highest_losses}

find_extreme_loss_ids()
