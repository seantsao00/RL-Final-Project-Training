from dataclasses import dataclass
from pathlib import Path

from transformers import AutoTokenizer, PreTrainedTokenizerBase, set_seed
from trl.scripts.utils import ScriptArguments, TrlParser
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.grpo_trainer import GRPOTrainer
from trl.trainer.model_config import ModelConfig
from trl.trainer.utils import get_peft_config

from .data import load_classeval_dataset_prompt_only
from .reward_classeval import (
    RewardConfig,
    create_reward_funcs,
)


@dataclass
class CustomArguments:
    dataset_train_split_end: int | None = None


def main(
    script_args: ScriptArguments,
    training_args: GRPOConfig,
    model_args: ModelConfig,
    reward_cfg: RewardConfig,
    custom_args: CustomArguments,
):
    set_seed(training_args.seed)

    print("Loading Classeval compositional dataset")
    train_dataset = load_classeval_dataset_prompt_only(
        0, custom_args.dataset_train_split_end
    )
    eval_dataset = load_classeval_dataset_prompt_only(
        custom_args.dataset_train_split_end, 100
    )
    
    print("train_dataset size:", len(train_dataset))
    print("eval_dataset size:", len(eval_dataset))

    training_args.reward_weights = [
        reward_cfg.tests_weight,
        reward_cfg.ruff_weight,
        reward_cfg.mypy_weight,
    ]

    reward_funcs = create_reward_funcs(reward_cfg, custom_args.test_threads)

    if model_args.model_name_or_path is None:
        raise ValueError("Model name or path must be specified in model_args.")
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path
    )
    tokenizer.padding_side = "left"
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_args),
    )

    print(f"Starting GRPO training for {training_args.num_train_epochs} epochs")

    trainer.train()

    if training_args.output_dir is None:
        raise ValueError("Output directory must be specified in training_args.")
    output_dir = Path(training_args.output_dir) / "grpo-final"
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    print(f"Model and tokenizer saved to {output_dir}")


if __name__ == "__main__":
    parser = TrlParser(
        (ScriptArguments, GRPOConfig, ModelConfig, RewardConfig, CustomArguments)  # type: ignore
    )
    script_args, training_args, model_args, reward_cfg, custom_args = (
        parser.parse_args_and_config()
    )
    main(script_args, training_args, model_args, reward_cfg, custom_args)
