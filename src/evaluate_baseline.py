"""
Evaluation script for the baseline Qwen2.5-Coder-3B-Instruct model on ClassEval holistic dataset.
This allows comparison between the original model and the GRPO-trained model.
"""
import argparse
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from datasets import Dataset

from .holisitc_classeval.data import load_classeval_holistic_dataset_prompt_only
from .holisitc_classeval.env import evaluate_unit_tests
from .reward import ruff_reward_function, mypy_reward_function
import re


@dataclass
class EvaluationMetrics:
    """Metrics for a single example evaluation."""
    test_passed: int
    test_total: int
    test_pass_rate: float
    syntax_error: bool
    ruff_score: float
    mypy_score: float
    overall_score: float
    stderr: str = ""


@dataclass
class AggregatedMetrics:
    """Aggregated metrics across all examples."""
    num_examples: int
    avg_test_pass_rate: float
    avg_ruff_score: float
    avg_mypy_score: float
    avg_overall_score: float
    syntax_error_count: int
    syntax_error_rate: float
    perfect_solutions: int  # All tests passed
    perfect_solution_rate: float


def _extract_code(completion: str) -> str:
    """Extract code from markdown code block if present."""
    match = re.search(r"```python(.*?)```", completion, re.DOTALL)
    return match.group(1).strip() if match else completion.strip()


def generate_completion(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: list[dict[str, str]],
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> str:
    """Generate a completion for the given prompt."""
    # Apply chat template
    formatted_prompt = tokenizer.apply_chat_template(
        prompt,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    
    # Decode only the new tokens (exclude the prompt)
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    completion = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return completion


def evaluate_single_example(
    completion: str,
    tests: str,
    tests_weight: float = 1.0,
    ruff_weight: float = 0.5,
    mypy_weight: float = 1.5,
) -> EvaluationMetrics:
    """Evaluate a single generated solution."""
    solution = _extract_code(completion)
    
    # Evaluate unit tests
    test_code = solution + '\n' + tests
    test_result = evaluate_unit_tests(test_code)
    
    if test_result.syntax_error:
        return EvaluationMetrics(
            test_passed=0,
            test_total=test_result.n_total,
            test_pass_rate=0.0,
            syntax_error=True,
            ruff_score=0.0,
            mypy_score=0.0,
            overall_score=0.0,
            stderr=test_result.stderr,
        )
    
    test_pass_rate = test_result.n_passed / test_result.n_total if test_result.n_total > 0 else 0.0
    
    # Evaluate ruff (code quality)
    ruff_scores = ruff_reward_function(
        prompts=[[]],  # Not used
        completions=[[{"role": "assistant", "content": completion}]],
    )
    ruff_score = ruff_scores[0]
    
    # Evaluate mypy (type checking)
    mypy_scores = mypy_reward_function(
        prompts=[[]],  # Not used
        completions=[[{"role": "assistant", "content": completion}]],
    )
    mypy_score = mypy_scores[0]
    
    # Calculate overall score (weighted average)
    total_weight = tests_weight + ruff_weight + mypy_weight
    overall_score = (
        test_pass_rate * tests_weight +
        ruff_score * ruff_weight +
        mypy_score * mypy_weight
    ) / total_weight
    
    return EvaluationMetrics(
        test_passed=test_result.n_passed,
        test_total=test_result.n_total,
        test_pass_rate=test_pass_rate,
        syntax_error=False,
        ruff_score=ruff_score,
        mypy_score=mypy_score,
        overall_score=overall_score,
        stderr=test_result.stderr,
    )


def evaluate_model(
    model_name: str,
    dataset: Dataset,
    output_file: Path | None = None,
    max_examples: int | None = None,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    tests_weight: float = 1.0,
    ruff_weight: float = 0.5,
    mypy_weight: float = 1.5,
) -> tuple[AggregatedMetrics, list[dict[str, Any]]]:
    """
    Evaluate the model on the dataset.
    
    Returns:
        - Aggregated metrics
        - List of detailed results for each example
    """
    print(f"Loading baseline model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.eval()
    
    # Limit dataset size if specified
    if max_examples is not None:
        dataset = dataset.select(range(min(len(dataset), max_examples)))
    
    print(f"Evaluating {len(dataset)} examples...")
    
    all_metrics = []
    detailed_results = []
    
    for idx, example in enumerate(tqdm(dataset, desc="Evaluating")):
        prompt = example["prompt"]
        tests = example["tests"]
        
        # Generate completion
        completion = generate_completion(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        
        # Evaluate
        metrics = evaluate_single_example(
            completion=completion,
            tests=tests,
            tests_weight=tests_weight,
            ruff_weight=ruff_weight,
            mypy_weight=mypy_weight,
        )
        
        all_metrics.append(metrics)
        
        # Store detailed result
        detailed_results.append({
            "example_id": idx,
            "prompt": prompt,
            "completion": completion,
            "tests": tests,
            "metrics": asdict(metrics),
        })
        
        # Print progress every 10 examples
        if (idx + 1) % 10 == 0:
            avg_pass_rate = sum(m.test_pass_rate for m in all_metrics) / len(all_metrics)
            print(f"Progress: {idx + 1}/{len(dataset)} - Avg pass rate: {avg_pass_rate:.2%}")
    
    # Calculate aggregated metrics
    num_examples = len(all_metrics)
    avg_test_pass_rate = sum(m.test_pass_rate for m in all_metrics) / num_examples
    avg_ruff_score = sum(m.ruff_score for m in all_metrics) / num_examples
    avg_mypy_score = sum(m.mypy_score for m in all_metrics) / num_examples
    avg_overall_score = sum(m.overall_score for m in all_metrics) / num_examples
    syntax_error_count = sum(1 for m in all_metrics if m.syntax_error)
    syntax_error_rate = syntax_error_count / num_examples
    perfect_solutions = sum(1 for m in all_metrics if m.test_pass_rate == 1.0 and not m.syntax_error)
    perfect_solution_rate = perfect_solutions / num_examples
    
    aggregated = AggregatedMetrics(
        num_examples=num_examples,
        avg_test_pass_rate=avg_test_pass_rate,
        avg_ruff_score=avg_ruff_score,
        avg_mypy_score=avg_mypy_score,
        avg_overall_score=avg_overall_score,
        syntax_error_count=syntax_error_count,
        syntax_error_rate=syntax_error_rate,
        perfect_solutions=perfect_solutions,
        perfect_solution_rate=perfect_solution_rate,
    )
    
    # Save results if output file specified
    if output_file is not None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        results = {
            "model_name": model_name,
            "model_type": "baseline",
            "aggregated_metrics": asdict(aggregated),
            "detailed_results": detailed_results,
        }
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_file}")
    
    return aggregated, detailed_results


def print_metrics(metrics: AggregatedMetrics, model_name: str):
    """Pretty print the aggregated metrics."""
    print("\n" + "=" * 60)
    print(f"BASELINE EVALUATION RESULTS: {model_name}")
    print("=" * 60)
    print(f"Total examples evaluated: {metrics.num_examples}")
    print(f"Average test pass rate: {metrics.avg_test_pass_rate:.2%}")
    print(f"Average ruff score: {metrics.avg_ruff_score:.4f}")
    print(f"Average mypy score: {metrics.avg_mypy_score:.4f}")
    print(f"Average overall score: {metrics.avg_overall_score:.4f}")
    print(f"Syntax errors: {metrics.syntax_error_count} ({metrics.syntax_error_rate:.2%})")
    print(f"Perfect solutions: {metrics.perfect_solutions} ({metrics.perfect_solution_rate:.2%})")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate baseline Qwen2.5-Coder model on ClassEval holistic dataset"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-Coder-3B-Instruct",
        help="HuggingFace model name (default: Qwen/Qwen2.5-Coder-3B-Instruct)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./evaluation_results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Maximum number of examples to evaluate (default: all)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=1000,
        help="Maximum samples from dataset before train/test split",
    )
    parser.add_argument(
        "--train_test_split_ratio",
        type=float,
        default=0.8,
        help="Train/test split ratio (default: 0.8)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for generation",
    )
    parser.add_argument(
        "--tests_weight",
        type=float,
        default=1.0,
        help="Weight for unit test reward",
    )
    parser.add_argument(
        "--ruff_weight",
        type=float,
        default=0.5,
        help="Weight for ruff reward",
    )
    parser.add_argument(
        "--mypy_weight",
        type=float,
        default=1.5,
        help="Weight for mypy reward",
    )
    
    args = parser.parse_args()
    
    # Load test dataset
    print("Loading test dataset...")
    test_dataset = load_classeval_holistic_dataset_prompt_only(
        split="test",
        max_samples=args.max_samples,
        train_test_split_ratio=args.train_test_split_ratio,
    )
    
    # Create output file path
    model_short_name = args.model_name.replace("/", "_")
    output_file = Path(args.output_dir) / f"{model_short_name}_baseline_evaluation.json"
    
    # Run evaluation
    metrics, detailed_results = evaluate_model(
        model_name=args.model_name,
        dataset=test_dataset,
        output_file=output_file,
        max_examples=args.max_examples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        tests_weight=args.tests_weight,
        ruff_weight=args.ruff_weight,
        mypy_weight=args.mypy_weight,
    )
    
    # Print results
    print_metrics(metrics, args.model_name)


if __name__ == "__main__":
    main()
