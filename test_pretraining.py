import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Union

from transformers import HfArgumentParser

from configs.config_maps import MODEL_PROTOTYPE_CONFIGS, TRAINING_CONFIGS

# add full path for test experiments
#python test_pretraining.py --job_dir=/home2/s3919609/pixel-semantic/scripts/pretraining/experiments   --prototype_config_name=scratch_noto_span0.25-dropout   --training_config_name=fp16_apex_bs32_multilingual_test

@dataclass
class TrainingArguments:
    job_dir: str = field(metadata={"help": "Job dir"})
    prototype_config_name: Optional[str] = field(
        default=None, metadata={"help": "Name of the model prototype config to train"}
    )
    training_config_name: Optional[str] = field(default=None, metadata={"help": "Name of the training config to use"})
    def __post_init__(self):
        if self.prototype_config_name and self.prototype_config_name not in MODEL_PROTOTYPE_CONFIGS:
            raise ValueError(
                f"Specified prototype model config not available. "
                f"Available options: {list(MODEL_PROTOTYPE_CONFIGS.keys())}"
            )

    def __str__(self):
        self_as_dict = asdict(self)
        attrs_as_str = [f"{k}={v},\n" for k, v in sorted(self_as_dict.items())]
        return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)})"

    __repr__ = __str__

def load_json(json_name: Union[str, os.PathLike]) -> Dict[str, Any]:
    with open(json_name, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def get_config_dict(args: TrainingArguments) -> Dict[str, Any]:
    config_dict = {"output_dir": args.job_dir, "run_name": "pixel-semantic"}

    # Model config
    if args.prototype_config_name:
        model_config = load_json(MODEL_PROTOTYPE_CONFIGS[args.prototype_config_name])
        config_dict.update(model_config)

    # Training config
    if args.training_config_name:
        training_config = load_json(TRAINING_CONFIGS[args.training_config_name])
        config_dict.update(training_config)

    print(config_dict)

    return config_dict

class Trainer:
    def __init__(self, config_dict: Dict[str, Any]):
        self.config_dict = config_dict

    def run(self):
        import scripts.training.run_pretraining_multilingual as trainer
        trainer.main(self.config_dict)

def main():
    parser = HfArgumentParser(TrainingArguments)
    args = parser.parse_args_into_dataclasses()[0]

    config_dict = get_config_dict(args)
    trainer = Trainer(config_dict)
    trainer.run()

if __name__ == "__main__":
    main()
