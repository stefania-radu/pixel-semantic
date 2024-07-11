# My semantic tasks

# change the cache so that we dont run out of space
export HF_HOME="/scratch/.cache/huggingface"
export HF_DATASETS_CACHE="/scratch/.cache/huggingface/datasets"
export TRANSFORMERS_CACHE="/scratch/.cache/huggingface/models"

# sequence classification - GLUE

# evaluation the model on COLA
# change task name in model path and task name to one of : 
# task_to_keys = {
#     "cola": ("sentence", None),
#     "mnli": ("premise", "hypothesis"),
#     "mrpc": ("sentence1", "sentence2"),
#     "qnli": ("question", "sentence"),
#     "qqp": ("question1", "question2"),
#     "rte": ("sentence1", "sentence2"),
#     "sst2": ("sentence", None),
#     "stsb": ("sentence1", "sentence2"),
#     "wnli": ("sentence1", "sentence2"),
# }




python scripts/training/run_glue.py   \
    --model_name_or_path="Team-PIXEL/pixel-base-finetuned-cola" \
    --task_name="cola" \
    --remove_unused_columns=False \
    --output_dir="sanity_check" \
    --do_eval \
    --max_seq_length=256 \
    --overwrite_cache


# NER - CoNLL2003 + MasakhaneNER (data is in /scratch/s3919609/data/masakhane-ner)

# evaluation the model on CoNLL2003
# change the model/data based on the language (conll_2003_en)
python scripts/training/run_ner.py \
    --model_name_or_path="Team-PIXEL/pixel-base-finetuned-masakhaner-amh" \
    --data_dir /scratch/s3919609/data/masakhane-ner/data/amh \
    --remove_unused_columns=False \
    --output_dir="test_ner" \
    --do_predict \
    --do_calibrate \
    --max_seq_length=256 

# Questions Answering - SQuAD (there is also tydiqa, korquad and jaquad)

# this takes 45 min so i wont run it;(

# change the dataname_dir based on the dataset. this is used by the load dataset function

python scripts/training/run_qa.py \
    --model_name_or_path="Team-PIXEL/pixel-base-finetuned-squadv1" \
    --dataset_name="squad"  \
    --remove_unused_columns=False  \
    --output_dir="sanity_check_qa"  \
    --do_eval  \
    --max_seq_length=256

