# Monte Carlo experiments

## Experiment 1: Uncertainty (SD) across tasks and languages

mask ratio = 0.25

To run this, I had to remove references to the pangocairo rederer because the installation did \
not work on Windows. I had to comment the lines:
- from .datasets import * from src\pixel\data\__init__.py
- from .question_answering import * : src\pixel\utils\__init__.py
- from .pangocairo_renderer import * : src\pixel\data\\rendering\__init__.py


Run:

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

Inputs:
- .json file containing the text data per id, per language, per task

Outputs:
- .json file with the mean loss after Monte Carlo (per id, language, task)
- .json file with the mean uncertainty (SD) after Monte Carlo (per id, language, task)
- plots with the original, original + SD and predictions + SD for the examples with the lowest and highest loss values (5 for each)

## Experiment 2: The Attention mechanism in the PIXEL model

Inputs:
- the worst and best predictions w.r.t the loss from the previous experiment
- (predicitons with the lowest and highest uncertainty?)

Outputs:
- attention grid like bertviz and close-ups (specific layer/head) for the worst and best predictions w.r.t the loss

Run: run script from before with do-attention flag

## Experiment 3: Mask ratio and uncertainty

Vary the mask ratio between 0.1 and 0.9. The rest stay the same.

Inputs:
- same as experiment 1, run the same script for each mask value then aggregate results

Outputs:
- line plot for loss in terms of mask ratio, where each line is a language/task
- line plot for SD in terms of mask ratio, where each line is a language/task
- (maybe show the triplets for one example for each mask ratio value)

## Experiment 4: Mask span and uncertainty

Vary the masking_max_span_length between 1 and 6. 

Inputs:
- same as experiment 1, run the same script for each mask value then aggregate results

Outputs:
- line plot for loss in terms of mask ratio, where each line is a language/task
- line plot for SD in terms of mask ratio, where each line is a language/task
- (maybe show the triplets for one example for each mask ratio value)