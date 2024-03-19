# Run Monte Carlo experiments

## Setup 1: Uncertainty across tasks and languages (+ attention)

mask ratio = 0.25

Run:

```bash
python scripts\monte_carlo\monte_carlo_experiments.py
  --model_name_or_path="Team-PIXEL/pixel-base" \
  --experiment_type="mask_ratio"
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
- attention grid and close-ups for the worst and best predictions w.r.t the loss

## Setup 2: Mask ratio experiment

Vary the mask ratio between 0.1 and 0.9. The rest stay the same.

## Setup 3: Mask span experiment

Vary the masking_max_span_length between 1 and 6. 