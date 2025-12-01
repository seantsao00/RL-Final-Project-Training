import re
from dataclasses import dataclass
from pathlib import Path

from transformers import AutoTokenizer, PreTrainedTokenizerBase, set_seed
from trl.scripts.utils import ScriptArguments, TrlParser
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.grpo_trainer import GRPOTrainer
from trl.trainer.model_config import ModelConfig
from trl.trainer.utils import get_peft_config

from .data import AppsSample, load_apps_dataset_prompt_only
from .env import evaluate_candidate
from .reward import RewardConfig, compute_reward


@dataclass
class CustomArguments:
    dataset_train_max_samples: int | None = None
    test_threads: int | None = None


def main(
    script_args: ScriptArguments,
    training_args: GRPOConfig,
    model_args: ModelConfig,
    reward_cfg: RewardConfig,
    custom_args: CustomArguments,
):
    set_seed(training_args.seed)

    print(f"Loading dataset: {script_args.dataset_train_split}")
    train_dataset = load_apps_dataset_prompt_only(
        script_args.dataset_train_split, custom_args.dataset_train_max_samples
    )
    eval_dataset = load_apps_dataset_prompt_only(
        script_args.dataset_test_split, custom_args.dataset_train_max_samples
    )

    def reward_function(
        prompts: list[list[dict[str, str]]],
        completions: list[list[dict[str, str]]],
        **kwargs,
    ) -> list[float]:
        rewards: list[float] = []

        questions = [prompt[1]["content"] for prompt in prompts]
        solutions = [prompt[0]["content"] for prompt in completions]
        tests: list[list[tuple[str, str]]] = kwargs["tests"]

        for i, (question, solution, test_cases) in enumerate(
            zip(questions, solutions, tests, strict=True)
        ):
            markdown_code_block = False
            match = re.search(r"```python(.*?)```", solution, re.DOTALL)
            if match:
                solution = match.group(1).strip()
                markdown_code_block = True

            sample = AppsSample(question=question, tests=test_cases)
            try:
                exec_res, ruff_res, mypy_res = evaluate_candidate(
                    solution, sample, custom_args.test_threads
                )
                reward_value = compute_reward(
                    markdown_code_block, exec_res, ruff_res, mypy_res, reward_cfg
                )
                if i == 0:
                    print("Debug info from reward_function for first sample:")
                    print(f"Question: {question}")
                    print("================================")
                    print("Generated Solution:")
                    print(solution)
                    print("================================")
                    print(f"Markdown Code Block: {markdown_code_block}")
                    print(f"Execution Result: {exec_res}")
                    print(f"Ruff Result: {ruff_res}")
                    print(f"Mypy Result: {mypy_res}")
                    print(f"Computed Reward: {reward_value}")
            except Exception as e:
                print(f"Evaluation error: {e}")
                reward_value = -1.0

            rewards.append(reward_value)

        return rewards

    if model_args.model_name_or_path is None:
        raise ValueError("Model name or path must be specified in model_args.")
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path
    )
    tokenizer.padding_side = "left"
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_function,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_args),
        processing_class=tokenizer,
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
