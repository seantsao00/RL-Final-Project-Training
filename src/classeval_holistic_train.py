from dataclasses import dataclass
from pathlib import Path

from transformers import AutoTokenizer, PreTrainedTokenizerBase, set_seed
from trl.scripts.utils import ScriptArguments, TrlParser
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.grpo_trainer import GRPOTrainer
from trl.trainer.model_config import ModelConfig
from trl.trainer.utils import get_peft_config

from .holisitc_classeval.data import load_classeval_holistic_dataset_prompt_only
from .reward import (
    RewardConfig,
    mypy_reward_function,
    ruff_reward_function,
    ablation_mypy_reward_function,
    ablation_ruff_reward_function
)
from .holisitc_classeval.reward import unit_test_reward_function


@dataclass
class CustomArguments:
    dataset_train_max_samples: int | None = None
    test_threads: int | None = None
    train_test_split_ratio: float = 0.8
    ablation: int = 0


def main(
    script_args: ScriptArguments,
    training_args: GRPOConfig,
    model_args: ModelConfig,
    reward_cfg: RewardConfig,
    custom_args: CustomArguments,
):
    set_seed(training_args.seed)

    print(f"Loading dataset with train/test split ratio: {custom_args.train_test_split_ratio}")
    train_dataset = load_classeval_holistic_dataset_prompt_only(
        script_args.dataset_train_split, 
        custom_args.dataset_train_max_samples,
        custom_args.train_test_split_ratio
    )
    eval_dataset = load_classeval_holistic_dataset_prompt_only(
        script_args.dataset_test_split, 
        custom_args.dataset_train_max_samples,
        custom_args.train_test_split_ratio
    )

    training_args.reward_weights = [
        reward_cfg.tests_weight,
        reward_cfg.ruff_weight,
        reward_cfg.mypy_weight,
    ]

    def wrapped_unit_test_reward_function(*args, **kwargs):
        return unit_test_reward_function(
            *args,
            syntax_error_penalty=reward_cfg.syntax_error_penalty,
            test_threads=custom_args.test_threads,
            **kwargs,
        )
    # def wrapped_ruff_reward_function(*args, **kwargs):
    #     return ruff_reward_function(
    #         *args,
    #         ruff_select=reward_cfg.ruff_select,
    #         ruff_ignore=reward_cfg.ruff_ignore,
    #         **kwargs,
    #     )

    if custom_args.ablation == 1:
        reward_funcs = [
            wrapped_unit_test_reward_function,
            ablation_ruff_reward_function,
            ablation_mypy_reward_function,
        ]
    else:
        reward_funcs = [
            wrapped_unit_test_reward_function,
            ruff_reward_function,
            mypy_reward_function,
        ]

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
    output_dir = Path(training_args.output_dir) / "grpo-classeval-holistic"
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
