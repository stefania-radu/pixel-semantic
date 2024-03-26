# Monte Carlo experiments

## Experiment 1: Uncertainty (SD) across tasks and languages

mask ratio = 0.25

To run this, I had to remove references to the pangocairo rederer because the installation did \
not work on Windows. I had to comment the lines:
- from .datasets import * from src\pixel\data\__init__.py
- from .question_answering import * : src\pixel\utils\__init__.py
- from .pangocairo_renderer import * : src\pixel\data\\rendering\__init__.py


To get json files:

```bash
python scripts\monte_carlo\monte_carlo_experiments.py \
  --model_name_or_path="Team-PIXEL/pixel-base" \
  --experiment_type="mask_ratio" \
  --do_loss \
  --do_std \
  --do_attention \
  --mask_ratio=0.25 \
  --masking_max_span_length=6 \
  --masking_cumulative_span_weights= "0.2,0.4,0.6,0.8,0.9,1"\
  --span_mask \
  --max_seq_length=256 \
```

To get the lowest, highest plots:

```bash
python scripts/monte_carlo/plot_monte_carlo.py
```

Inputs:
- [.json file](scripts/data/uncertainty/test_data_ner_tydiqa_glue_small.json) containing the 290 examples of text data per id, per language, per task. There are 10 examples per language (in the case of GLUE, this is done per subtask)

Outputs:
- [loss_per_task_mask_0.25](results/base_experiment/loss_per_task_mask_0.25.json) .json file with the mean loss after Monte Carlo (per id, language, task)
- [SD_per_task_mask_0.25](results/base_experiment/SD_per_task_mask_0.25.json) .json file with the mean uncertainty (SD) after Monte Carlo (per id, language, task)
- [plots](results/base_experiment/images) with the original, original + SD and predictions + SD for the examples with the lowest and highest loss values (5 for each)

<p align="middle">
 <img src="results/base_experiment/images/base_lowest.png" width="400" height="600"></img>
  <img src="results/base_experiment/images/base_highest.png" width="400" height="600"></img>
</p>

<!-- <p align="middle">
 <img src="results/base_experiment/images/base_highest.png" width="400" height="600"></img>
</p> -->

## Experiment 2: The Attention mechanism in the PIXEL model

Inputs:
- the worst and best predictions w.r.t the loss from the previous experiment
- (predicitons with the lowest and highest uncertainty?)

Outputs:
- attention grid like bertviz and close-ups (specific layer/head) for the worst and best predictions w.r.t the loss


<p align="middle">
 <img src="results/attention/attention-english.png" width="800" height="400"></img>
</p>


<p align="middle">
 <img src="results/attention/attention-korean.png" width="800" height="400"></img>
</p>

Run: run script from before with do-attention flag

## Experiment 3: Mask ratio and uncertainty

Vary the mask ratio between 0.1 and 0.9. The rest stays the same.

Example run for mask_ratio = 0.9. The experiment type is mask_ratio.

```bash
python scripts/monte_carlo/monte_carlo_experiments.py \
  --model_name_or_path="Team-PIXEL/pixel-base" \
  --experiment_type="mask_ratio" \
  --do_loss \
  --do_std \
  --mask_ratio=0.9 \
  --masking_spacing=0 \
  --masking_max_span_length=6 \
  --masking_cumulative_span_weights="0.2,0.4,0.6,0.8,0.9,1" \
  --span_mask \
  --max_seq_length=256 \
```

Inputs:
- same as experiment 1, run the same script for each mask value then aggregate results

Outputs:
- line plot for loss in terms of mask ratio, where each line is a language/task
- line plot for SD in terms of mask ratio, where each line is a language/task
- (maybe show the triplets for one example for each mask ratio value)

Note: for a high mask_ratio (> 0.8), the model with cap it at 0.5 because some images have very little text

## Experiment 4: Mask span and uncertainty

Vary the masking_max_span_length between 1 and 6 and keep the probability to 1.

Example run for span=6. There will only be sequences of 6 consecutive patches with a probability of 100%. The experiment type is span.


```bash
python scripts/monte_carlo/monte_carlo_experiments.py \
  --model_name_or_path="Team-PIXEL/pixel-base" \
  --experiment_type="span" \
  --do_loss \
  --do_std \
  --mask_ratio=0.25 \
  --masking_max_span_length=6 \
  --masking_cumulative_span_weights="0, 0, 0, 0, 0, 1" \
  --span_mask \
  --max_seq_length=256 \
```

Inputs:
- same as experiment 1, run the same script for each mask value then aggregate results

Outputs:
- line plot for loss in terms of mask ratio, where each line is a language/task
- line plot for SD in terms of mask ratio, where each line is a language/task
- (maybe show the triplets for one example for each mask ratio value)