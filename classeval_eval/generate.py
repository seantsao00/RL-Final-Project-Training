import re
from dataclasses import dataclass
from pathlib import Path

import torch
from peft import PeftModel
from src.data import load_classeval_dataset_prompt_only
from src.env_classeval import build_full_class_code, add_timeout_to_unittest_code
from src.reward_classeval import RewardConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, GenerationConfig
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
        Path(__file__).resolve().parents[2]
        / "ClassEval"
        / "data"
        / "ClassEval_data.json"
    ).as_posix()
    greedy: int = 1
    eval_output_dir: str = "classeval_eval/results/eval_result"
    lora_weights: str | None = None
    test_work: bool = False


def _extract_code(completion: str) -> str:
    """Extract code from markdown code block if present."""
    pattern_list = [
        r"```python(.*?)```",
        r"```ruby(.*?)```",
        r"```scss(.*?)```",
        r"```python(.*?)",
        r"```(.*?)```",
        r"\[PYTHON\](.*?)\[/PYTHON\]",
    ]
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
    eval_output_dir = Path(eval_args.eval_output_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    set_seed(training_args.seed)
    eval_start = 99 if eval_args.test_work else custom_args.dataset_train_split_end
    eval_dataset = load_classeval_dataset_prompt_only(eval_start, 100)
    if eval_args.test_work:
        print("This is a test run. Only test on 1 class sample.")
    print("eval_dataset size:", len(eval_dataset))
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
    ).to(device)

    if eval_args.lora_weights:
        lora_path = Path(eval_args.lora_weights)
        if not lora_path.exists():
            raise FileNotFoundError(f"LoRA weights path '{lora_path}' does not exist.")
        model = PeftModel.from_pretrained(model, str(lora_path))
    else:
        print(
            "\033[33m[Warning]\033[0m No LoRA adapter provided (eval_args.lora_weights is None).\n"
            "Using the base model only. If you intended to evaluate a fine-tuned adapter, "
            "pass --eval_arguments.lora_weights or configure EvalArguments.lora_weights."
        )

    model.eval()

    greedy = bool(eval_args.greedy)

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

        generate_config = GenerationConfig(
            temperature=0 if greedy else training_args.temperature,
            top_p=training_args.top_p,
            top_k=training_args.top_k,
        )

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=training_args.max_completion_length,
                generation_config=generate_config,
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

    eval_output_dir.mkdir(parents=True, exist_ok=True)
    code_dir = eval_output_dir / "composed_code"
    code_dir.mkdir(parents=True, exist_ok=True)

    composed_files: list[Path] = []
    for class_name, class_code in tqdm(
        compositional_result.items(), desc="Writing composed class files"
    ):
        cls_dir = code_dir / class_name
        cls_dir.mkdir(parents=True, exist_ok=True)
        class_file = cls_dir / f"{class_name}.py"
        with class_file.open("w", encoding="utf-8") as f:
            f.write(class_code)
        composed_files.append(class_file)

    # Build full_test_code files per class here
    test_files: list[Path] = []
    for class_name, class_code in compositional_result.items():
        unittest_code = code_metadata[class_name]["class_test_code"]
        unittest_code = add_timeout_to_unittest_code(unittest_code)
        full_test_code = f"""
{class_code}

{unittest_code}
"""
        if "unittest.main()" not in full_test_code:
            full_test_code += """

if __name__ == "__main__":
    unittest.main()
"""

        full_test_file = code_dir / class_name / f"full_test_{class_name}.py"
        full_test_file.parent.mkdir(parents=True, exist_ok=True)
        with full_test_file.open("w", encoding="utf-8") as f:
            f.write(full_test_code)
        test_files.append(full_test_file)

    summary_file = eval_output_dir / "results_summary.json"
    results_summary = run_evaluation_and_save(
        test_files=test_files,
        composed_files=composed_files,
        summary_file=summary_file,
    )
    # Message moved to evaluate.py; still confirm code and summary locations
    print(f"Saved composed class code to {code_dir}")


if __name__ == "__main__":
    parser = TrlParser(
        (
            ScriptArguments,
            GRPOConfig,
            ModelConfig,
            RewardConfig,
            CustomArguments,
            EvalArguments,
        )
    )
    script_args, training_args, model_args, reward_cfg, custom_args, eval_args = (
        parser.parse_args_and_config()
    )
    main(script_args, training_args, model_args, reward_cfg, custom_args, eval_args)
