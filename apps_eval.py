
import argparse
import json
import os
import re

import torch
from datasets import Dataset, load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.env import ExecutionResult, evaluate_mypy, evaluate_ruff, evaluate_unit_tests


def get_system_prompt() -> str:
    return """You are Qwen, created by Alibaba Cloud. You are a helpful assistant.
You will be given a programming question and you must provide a solution in Python.
Your output must be only Python code, no explanations, no comments, no markdown.
The program must be a standalone solution using only the Python standard library.
The program should read input exactly as described.
The program should print only the required output.
"""


def get_user_prompt(question: str) -> str:
    return f"""Question:
{question}

Give a Python solution.
"""


def generate_solution(model, tokenizer, question: str) -> str:
    """Generate solution using the loaded model."""
    messages = [
        {"role": "system", "content": get_system_prompt()},
        {"role": "user", "content": get_user_prompt(question)},
    ]
    
    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the generated part
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return response


def evaluate_solution(
    code: str,
    inputs: list[str],
    outputs: list[str],
) -> ExecutionResult:
    tests = list(zip(inputs, outputs, strict=True))
    result = evaluate_unit_tests(code, tests, timeout_s=2.0)
    return result


def lint_solution(code: str) -> int:
    select = ["F", "E", "W", "C90", "N", "UP", "B", "A", "C4", "RET", "SIM", "ARG"]
    ignore = ["E501", "E741", "W292"]
    result = evaluate_ruff(code, select, ignore)

    print("Ruff issues:")
    for message in result.messages:
        print(message)

    return result.n_issues


def check_mypy(code: str) -> int:
    result = evaluate_mypy(code)

    print("Mypy issues:")
    for message in result.messages:
        print(message)

    return result.n_errors


def evaluate_model(
    model,
    tokenizer,
    dataset: Dataset,
    model_name: str,
    num_examples: int | None,
    output_file: str,
) -> dict:
    """Evaluate a single model and return results."""
    print(f"\n{'='*80}")
    print(f"Evaluating model: {model_name}")
    print(f"{'='*80}")

    # If num_examples is None, evaluate all problems
    num_to_evaluate = len(dataset) if num_examples is None else min(num_examples, len(dataset))
    print(f"Evaluating {num_to_evaluate} problems")

    total_ruff_errors = 0
    total_mypy_errors = 0
    total_ac_cnt = 0
    total_test_cases = 0
    syntax_error_cnt = 0
    results = []

    for i in range(num_to_evaluate):
        print(f"\n--- Problem {i + 1}/{num_to_evaluate} ---")
        problem = dataset[i]
        question: str = problem["question"]
        input_output = json.loads(problem["input_output"])
        inputs: list[str] = input_output["inputs"]
        outputs: list[str] = input_output["outputs"]

        print("Generating code...")
        code = generate_solution(model, tokenizer, question)

        # TODO: Enforce no markdown code blocks in prompt itself
        m = re.search(r"```(?:python)?\s*(.*?)```", code, re.S)
        if m:
            code = m.group(1).strip()

        print("Code generated:")
        print(code[:200] + "..." if len(code) > 200 else code)

        print("Evaluating correctness...")
        result = evaluate_solution(code, inputs, outputs)
        
        problem_result = {
            "problem_id": i,
            "question": question,
            "code": code,
            "syntax_error": result.syntax_error,
        }

        if result.syntax_error:
            print("Syntax error in generated code.")
            syntax_error_cnt += 1
            problem_result.update({
                "n_passed": 0,
                "n_total": len(inputs),
                "timed_out": False,
                "runtime_error": False,
                "ruff_errors": 0,
                "mypy_errors": 0,
            })
            results.append(problem_result)
            continue

        n_failed = result.n_total - result.n_passed
        print(
            f"Passed: {result.n_passed}/{result.n_total} test cases. "
            f"Failed: {n_failed}, Timed out: {result.timed_out}, Runtime error: {result.runtime_error}"
        )
        total_ac_cnt += result.n_passed
        total_test_cases += result.n_total

        print("Linting code with Ruff...")
        num_ruff_errors = lint_solution(code)
        print(f"Ruff errors: {num_ruff_errors}")
        total_ruff_errors += num_ruff_errors

        print("Checking types with Mypy...")
        num_mypy_errors = check_mypy(code)
        print(f"Mypy errors: {num_mypy_errors}")
        total_mypy_errors += num_mypy_errors

        problem_result.update({
            "n_passed": result.n_passed,
            "n_total": result.n_total,
            "timed_out": result.timed_out,
            "runtime_error": result.runtime_error,
            "ruff_errors": num_ruff_errors,
            "mypy_errors": num_mypy_errors,
        })
        results.append(problem_result)

    summary = {
        "model": model_name,
        "num_problems": num_to_evaluate,
        "total_ruff_errors": total_ruff_errors,
        "avg_ruff_errors": total_ruff_errors / num_to_evaluate,
        "total_mypy_errors": total_mypy_errors,
        "avg_mypy_errors": total_mypy_errors / num_to_evaluate,
        "syntax_error_count": syntax_error_cnt,
        "total_test_cases_passed": total_ac_cnt,
        "total_test_cases": total_test_cases,
        "pass_rate": total_ac_cnt / total_test_cases if total_test_cases > 0 else 0,
        "problems": results,
    }

    # Save results to file
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Summary for {model_name}")
    print(f"{'='*80}")
    print(f"Evaluated {num_to_evaluate} problems.")
    print(f"Total Ruff errors: {total_ruff_errors}")
    print(f"Avg. Ruff errors per solution: {total_ruff_errors / num_to_evaluate:.2f}")
    print(f"Total Mypy errors: {total_mypy_errors}")
    print(f"Avg. Mypy errors per solution: {total_mypy_errors / num_to_evaluate:.2f}")
    print(f"Solutions with syntax errors: {syntax_error_cnt}")
    print(
        f"Total test cases passed: {total_ac_cnt} out of {total_test_cases} "
        f"({total_ac_cnt / total_test_cases * 100:.2f}%)" if total_test_cases > 0 else "No test cases"
    )
    print(f"Results saved to: {output_file}")
    
    return summary


def load_model_and_tokenizer(model_path: str, is_checkpoint: bool = False):
    """Load model and tokenizer from path."""
    print(f"\nLoading model from: {model_path}")
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load tokenizer
    if is_checkpoint:
        # For checkpoint, load tokenizer from the base model
        base_model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
        print(f"Loading tokenizer from base model: {base_model_name}")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        # Load base model
        print(f"Loading base model: {base_model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        
        # Load LoRA adapter
        print(f"Loading LoRA adapter from: {model_path}")
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()  # Merge LoRA weights
    else:
        # Load full model directly
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
    
    model.eval()
    print("Model loaded successfully!")
    
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-examples",
        "-n",
        type=int,
        default=None,
        help="Number of problems from the dataset to evaluate (default: all problems).",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="apps_eval",
        help="Directory to save evaluation results.",
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=str,
        default="v4checkpoints/checkpoint-186",
        help="Path to the checkpoint directory to evaluate (default: v4checkpoints/checkpoint-186).",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Only evaluate the baseline model, skip checkpoint evaluation.",
    )
    parser.add_argument(
        "--checkpoint-only",
        action="store_true",
        help="Only evaluate the checkpoint model, skip baseline evaluation.",
    )
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading dataset: apps ...")
    dataset: Dataset = load_dataset("codeparrot/apps", split="test")  # type: ignore
    dataset = dataset.filter(lambda x: x["difficulty"] == "introductory")

    # Extract checkpoint name for output file
    checkpoint_name = os.path.basename(args.checkpoint.rstrip('/'))

    # Define models to evaluate: (model_path, is_checkpoint, output_name)
    models_to_evaluate = []
    
    if not args.checkpoint_only:
        models_to_evaluate.append(
            ("Qwen/Qwen2.5-Coder-7B-Instruct", False, "qwen2.5-coder-7b-baseline")
        )
    
    if not args.baseline_only:
        models_to_evaluate.append(
            (args.checkpoint, True, checkpoint_name)
        )

    all_results = {}
    
    for model_path, is_checkpoint, output_prefix in models_to_evaluate:
        output_file = os.path.join(args.output_dir, f"{output_prefix}_evaluation_results.json")
        
        try:
            # Load model
            model, tokenizer = load_model_and_tokenizer(model_path, is_checkpoint)
            
            # Evaluate model
            summary = evaluate_model(
                model, tokenizer, dataset, model_path, args.num_examples, output_file
            )
            all_results[output_prefix] = summary
            
            # Free up memory
            del model
            del tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"\n!!! Error evaluating model {model_path}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Print comparison summary
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}")
    for model_name, summary in all_results.items():
        print(f"\n{model_name}:")
        print(f"  Pass rate: {summary['pass_rate']*100:.2f}%")
        print(f"  Syntax errors: {summary['syntax_error_count']}/{summary['num_problems']}")
        print(f"  Avg Ruff errors: {summary['avg_ruff_errors']:.2f}")
        print(f"  Avg Mypy errors: {summary['avg_mypy_errors']:.2f}")


if __name__ == "__main__":
    main()