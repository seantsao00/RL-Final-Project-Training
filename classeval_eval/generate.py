import re
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch
from peft import PeftModel
from src.data import load_classeval_dataset_prompt_only
from src.env_classeval import build_full_class_code
from src.reward_classeval import RewardConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl.scripts.utils import ScriptArguments, TrlParser
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.model_config import ModelConfig

from .evaluate import run_evaluation_and_save


@dataclass
class CustomArguments:
    dataset_train_split_end: int | None = None

@dataclass
class EvalArguments:
    data_path: str = (
        Path(__file__).resolve().parents[2] / "ClassEval" / "data" / "ClassEval_data.json"
    ).as_posix()
    greedy: int = 1
    output_path: str = "gen_result"
    lora_weights: str | None = None
    test_work: bool = False


def _extract_code(completion: str) -> str:
    """Extract code from markdown code block if present."""
    pattern_list = [r"```python(.*?)```", r"```ruby(.*?)```", r"```scss(.*?)```",
                    r"```python(.*?)", r"```(.*?)```", r"\[PYTHON\](.*?)\[/PYTHON\]"]
    for pattern in pattern_list:
        try:
            code = re.findall(pattern, completion, re.S)[0]
            return code
        except:
            continue
    return completion


def main(
    script_args: ScriptArguments,
    training_args: GRPOConfig,
    model_args: ModelConfig,
    reward_cfg: RewardConfig,
    custom_args: CustomArguments,
    eval_args: EvalArguments,
):
    output_path = Path(eval_args.output_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    set_seed(training_args.seed)
    eval_dataset = load_classeval_dataset_prompt_only(
        custom_args.dataset_train_split_end, 100
    )
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
    ).to(device)

    temperature = training_args.temperature

    if eval_args.lora_weights:
        lora_path = Path(eval_args.lora_weights)
        if not lora_path.exists():
            raise FileNotFoundError(f"LoRA weights path '{lora_path}' does not exist.")
        model = PeftModel.from_pretrained(model, str(lora_path))

    model.eval()

    greedy = bool(eval_args.greedy)
    do_sample = not greedy

    generated_code: dict[str, dict[str, str]] = {}
    code_metadata: dict[str, dict[str, any]] = {}

    for row in tqdm(eval_dataset, desc="Generating methods (eval split)"):
        messages = row["prompt"]
        class_name = row["class_name"]
        if not generated_code.get(class_name):
            generated_code[class_name] = {}
            code_metadata[class_name] = row
        method_name = row["method_name"]
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=training_args.max_completion_length,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        completion = tokenizer.decode(
            outputs[0][input_ids.shape[-1] :], skip_special_tokens=True
        )
        code = _extract_code(completion)
        generated_code[class_name][method_name] = code

    compositional_result: dict[str, str] = {}
    for class_name, methods in generated_code.items():
        compositional_result[class_name] = build_full_class_code(
            class_name=class_name,
            import_statement=code_metadata[class_name]["import_statement"],
            class_description="",
            class_constructor=code_metadata[class_name]["class_constructor"],
            methods_info=code_metadata[class_name]["methods_info"],
            replaced_method=methods,
        )

    output_path.mkdir(parents=True, exist_ok=True)
    code_dir = output_path / "composed_code"
    code_dir.mkdir(parents=True, exist_ok=True)

    for class_name, class_code in tqdm(
        compositional_result.items(), desc="Writing composed class files"
    ):
        cls_dir = code_dir / class_name
        cls_dir.mkdir(parents=True, exist_ok=True)
        class_file = cls_dir / f"{class_name}.py"
        with class_file.open("w", encoding="utf-8") as f:
            f.write(class_code)

    summary_file = output_path / "results_summary.json"
    results_summary = run_evaluation_and_save(
        composed_classes=compositional_result,
        code_metadata=code_metadata,
        code_dir=code_dir,
        summary_file=summary_file,
    )

    # Compute and print passrate to stdout
    total = len(results_summary)
    passed = sum(1 for r in results_summary if r["n_passed"] == r["n_total"])

    passrate = (passed / total * 100.0) if total > 0 else 0.0
    print(f"Saved composed class code to {code_dir} and test summary to {summary_file}")
    print(f"Passrate: {passed}/{total} ({passrate:.2f}%)")


if __name__ == "__main__":
    parser = TrlParser(
        (ScriptArguments, GRPOConfig, ModelConfig, RewardConfig, CustomArguments, EvalArguments)
    )
    script_args, training_args, model_args, reward_cfg, custom_args, eval_args = (
        parser.parse_args_and_config()
    )
    main(script_args, training_args, model_args, reward_cfg, custom_args, eval_args)
