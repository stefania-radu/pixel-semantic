# Uncertainty in PIXEL Text Reconstruction



## The effect of mask ratio on uncertainty across datasets and languages

Vary the mask ratio between 0.1 and 0.9.

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
- [.json file](../data/uncertainty/test_data_ner_tydiqa_glue_small.json) containing the 290 examples of text data per id, per language, per task. There are 10 examples per language (in the case of GLUE, this is done per subtask)

[Outputs](results/mask_experiment):
- line plot for loss in terms of mask ratio, where each line is a language/dataset (mean values are computed)

<p align="middle">
 <img src="results/mask_experiment/Mask_Loss_tasks_line_plot.png" width="800" height="400"></img>
</p>


<p align="middle">
 <img src="results/mask_experiment/Mask_Loss_languages_line_plot.png" width="800" height="400"></img>
</p>


- line plot for uncertainty (SD) in terms of mask ratio, where each line is a language/dataset (mean values are computed)

<p align="middle">
 <img src="results/mask_experiment/Mask_Uncertainty_tasks_line_plot.png" width="800" height="400"></img>
</p>


<p align="middle">
 <img src="results/mask_experiment/Mask_Uncertainty_languages_line_plot.png" width="800" height="400"></img>
</p>

- violin plots for the loss/uncertainty across datasets (plot is too messy for languages)

<p align="middle">
 <img src="results/mask_experiment/Mask_Loss_tasks_violin_plot.png" width="800" height="400"></img>
</p>

<p align="middle">
 <img src="results/mask_experiment/Mask_Uncertainty_tasks_violin_plot.png" width="800" height="400"></img>
</p>

Note: for a high mask_ratio (> 0.8), the model will cap it at 0.5 because some images have very little text.

## The effect of span length on uncertainty across datasets and languages

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
- [.json file](../data/uncertainty/test_data_ner_tydiqa_glue_small.json) containing the 290 examples of text data per id, per language, per task. There are 10 examples per language (in the case of GLUE, this is done per subtask)

[Outputs](results/span_experiment):
- line plot for loss in terms of span length, where each line is a language/dataset (mean values are computed)

<p align="middle">
 <img src="results/span_experiment/Span_Loss_tasks_line_plot.png" width="800" height="400"></img>
</p>


<p align="middle">
 <img src="results/span_experiment/Span_Loss_languages_line_plot.png" width="800" height="400"></img>
</p>


- line plot for uncertainty (SD) in terms of span length, where each line is a language/dataset (mean values are computed)

<p align="middle">
 <img src="results/span_experiment/Span_Uncertainty_tasks_line_plot.png" width="800" height="400"></img>
</p>


<p align="middle">
 <img src="results/span_experiment/Span_Uncertainty_languages_line_plot.png" width="800" height="400"></img>
</p>

- violin plot for the loss/uncertainty across datasets (plot is too messy for languages)

<p align="middle">
 <img src="results/span_experiment/Span_Loss_tasks_violin_plot.png" width="800" height="400"></img>
</p>

<p align="middle">
 <img src="results/span_experiment/Span_Uncertainty_tasks_violin_plot.png" width="800" height="400"></img>
</p>

## Visualizing uncertainty (top 5 performers + top 5 challenges in terms of reconstruction loss)

mask ratio = 0.25


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

Images are shown in the increasing order of the loss for both plots.

<p align="middle">
 <img src="results/base_experiment/images/base_lowest.png" width="400" height="600"></img>
  <img src="results/base_experiment/images/base_highest.png" width="400" height="600"></img>
</p>

# Table

| Category   | Amharic | Arabic | Bengali | Chinese | CoLA | CoNLL 2003 English | English | Finnish | Hausa | Igbo | Indonesian | Kinyarwanda | Korean | Luganda | Luo | MNLI | MRPC | Nigerian Pidgin | QNLI | QQP | RTE | Russian | SST-2 | STS-B | Swahili (NER) | Swahili (TyDi QA) | Telugu | WNLI | Wolof | Yoruba |
|------------|---------|--------|---------|---------|------|--------------------|---------|---------|-------|------|------------|-------------|--------|---------|-----|------|------|-----------------|------|-----|-----|---------|-------|-------|---------------|-------------------|--------|------|-------|--------|
| Mean Loss  | 0.83    | 0.83   | 0.69    | 0.85    | 0.47 | 0.69               | 0.22    | 0.54    | 0.62  | 0.62 | 0.52       | 0.60        | 0.85   | 0.60    | 0.58| 0.23 | 0.29 | 0.56            | 0.22 | 0.35| 0.21| 0.63    | 0.48  | 0.24  | 0.59          | 0.51              | 0.77   | 0.18 | 0.68  | 0.65   |
| Mean SD    | 0.15    | 0.05   | 0.10    | 0.14    | 0.06 | 0.10               | 0.03    | 0.04    | 0.04  | 0.04 | 0.02       | 0.04        | 0.08   | 0.04    | 0.04| 0.05 | 0.06 | 0.04            | 0.05 | 0.04| 0.04| 0.03    | 0.05  | 0.05  | 0.04          | 0.03              | 0.04   | 0.05 | 0.05  | 0.09   |





## The Attention mechanism in the PIXEL model

Inputs:
- the worst and best predictions w.r.t the loss from the previous experiment
- (predicitons with the lowest and highest uncertainty?)

Outputs:
- attention grid like bertviz and close-ups (specific layer/head) for the worst and best predictions w.r.t the loss


<p align="middle">
 <img src="results/attention/attention-english.png" width="800" height="600"></img>
</p>


<p align="middle">
 <img src="results/attention/attention-korean.png" width="800" height="600"></img>
</p>

Run: run script from before with do-attention flag



## Data

| Datasets | Total Examples | Languages/Subtasks | Examples |
|----------|----------------|--------------------|-------------------------------|
| NER      | 120 examples | amh | 10 |
|          |                                | conll_2003_en | 10 |
|          |                                | hau | 10 |
|          |                                | ibo | 10 |
|          |                                | kin | 10 |
|          |                                | lug | 10 |
|          |                                | luo | 10 |
|          |                                | pcm | 10 |
|          |                                | swa | 10 |
|          |                                | wol | 10 |
|          |                                | yor | 10 |
|          |                                | zh | 10 |
| Tydiqa   | 90 examples    | arabic | 10 |
|          |                                | russian | 10 |
|          |                                | bengali | 10 |
|          |                                | telugu | 10 |
|          |                                | finnish | 10 |
|          |                                | swahili | 10 |
|          |                                | korean | 10 |
|          |                                | indonesian | 10 |
|          |                                | english | 10 |
| GLUE     | 80 examples | cola | 10 |
|          |                                                         | mnli | 10 |
|          |                                                         | mrpc | 10 |
|          |                                                         | qnli | 10 |
|          |                                                         | qqp | 10 |
|          |                                                         | rte | 10 |
|          |                                                         | sst2 | 10 |
|          |                                                         | stsb | 10 |
|          |                                                         | wnli | 10 |
