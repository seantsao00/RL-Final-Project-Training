import re
import argparse
import json
from pathlib import Path
import tempfile
from typing import Dict, Any
import os
from contextlib import contextmanager

from src.data import load_classeval_dataset_prompt_only
from src.env_classeval import build_full_class_code, ClassEvalExecutionResult
from .evaluate import run_evaluation_and_save
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
)
from peft import PeftModel
import torch
import yaml
import subprocess
import shutil


@contextmanager
def _temp_code_file(code: str):
    """Context manager that creates a temporary Python file with the given code."""
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        candidate_path = tmp_dir / "candidate.py"
        candidate_path.write_text(code)
        yield candidate_path


def _extract_code(completion: str) -> str:
    """Extract code from markdown code block if present."""
    match = re.search(r"```python(.*?)```", completion, re.DOTALL)
    return match.group(1).strip() if match else completion

# Evaluation functions moved to generation/evaluate.py


def args_init():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/3b_lora_classeval.yaml",
        help="YAML config path to align inference with training settings",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="../data/ClassEval_data.json",
        help="ClassEval data",
    )
    parser.add_argument(
        "--greedy",
        type=int,
        default=1,
        help="Whether to generate with greedy decoding (1) or sampling (0)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="gen_result",
    )
    parser.add_argument(
        "--lora_weights",
        type=str,
        help="lora weights path",
    )
    parser.add_argument(
        "--test_work",
        action="store_true",
        default=False,
        help="Whether to do a quick test run",
    )
    return parser


def main():
    args = args_init().parse_args()
    output_path = Path(args.output_path)

    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    set_seed(cfg["seed"])
    base_model = cfg["model_name_or_path"]
    temperature = cfg["temperature"]
    max_length = cfg["max_completion_length"]
    use_peft = cfg["use_peft"]
    eval_start = cfg["dataset_train_split_end"] if not args.test_work else 99
    eval_end = 100

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.padding_side = "left"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    eval_dataset = load_classeval_dataset_prompt_only(eval_start, eval_end)

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
    ).to(device)

    if args.lora_weights:
        lora_path = Path(args.lora_weights)
        if not lora_path.exists():
            raise FileNotFoundError(f"LoRA weights path '{lora_path}' does not exist.")
        model = PeftModel.from_pretrained(model, str(lora_path))

    model.eval()

    results: list[Dict[str, Any]] = []

    greedy = bool(args.greedy)
    do_sample = not greedy

    generated_code: dict[str, dict[str, str]] = {}
    code_metadata: dict[str, Dict[str, Any]] = {}

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
                max_new_tokens=max_length,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                # top_p=0.9 if do_sample else None,
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
    main()
