"""
Script to manually input model completions and evaluate them in the same JSON format.
Useful for testing specific classes like Thermostat with different model outputs.
"""
import argparse
import json
from pathlib import Path
from dataclasses import asdict
from typing import Any

from datasets import load_dataset

from src.holisitc_classeval.data import question_to_prompt
from src.classeval_holistic_evaluate import (
    evaluate_single_example,
    EvaluationMetrics,
)


def load_class_data(class_name: str) -> dict[str, Any]:
    """Load the prompt and tests for a specific class from the dataset."""
    dataset = load_dataset(
        "FudanSELab/ClassEval", split="test", trust_remote_code=True
    )
    
    for row in dataset:
        if row["class_name"] == class_name:
            prompt = question_to_prompt(row["class_name"], row["skeleton"])
            return {
                "class_name": class_name,
                "prompt": prompt,
                "tests": row["test"],
                "skeleton": row["skeleton"],
            }
    
    raise ValueError(f"Class '{class_name}' not found in dataset")


def evaluate_manual_completion(
    class_name: str,
    completion: str,
    tests_weight: float = 1.0,
    ruff_weight: float = 0.5,
    mypy_weight: float = 1.5,
) -> tuple[dict[str, Any], EvaluationMetrics]:
    """Evaluate a manually provided completion."""
    class_data = load_class_data(class_name)
    
    # Evaluate the completion
    metrics = evaluate_single_example(
        completion=completion,
        tests=class_data["tests"],
        tests_weight=tests_weight,
        ruff_weight=ruff_weight,
        mypy_weight=mypy_weight,
    )
    
    result = {
        "example_id": 0,
        "class_name": class_name,
        "prompt": class_data["prompt"],
        "completion": completion,
        "tests": class_data["tests"],
        "metrics": asdict(metrics),
    }
    
    return result, metrics


def save_evaluation_result(
    model_name: str,
    results: list[dict[str, Any]],
    output_dir: Path,
):
    """Save evaluation results in the standard format."""
    # Calculate aggregated metrics
    all_metrics = [result["metrics"] for result in results]
    num_examples = len(all_metrics)
    
    avg_test_pass_rate = sum(m["test_pass_rate"] for m in all_metrics) / num_examples
    avg_ruff_score = sum(m["ruff_score"] for m in all_metrics) / num_examples
    avg_mypy_score = sum(m["mypy_score"] for m in all_metrics) / num_examples
    avg_overall_score = sum(m["overall_score"] for m in all_metrics) / num_examples
    syntax_error_count = sum(1 for m in all_metrics if m["syntax_error"])
    syntax_error_rate = syntax_error_count / num_examples
    perfect_solutions = sum(
        1 for m in all_metrics 
        if m["test_pass_rate"] == 1.0 and not m["syntax_error"]
    )
    perfect_solution_rate = perfect_solutions / num_examples
    
    aggregated_metrics = {
        "num_examples": num_examples,
        "avg_test_pass_rate": avg_test_pass_rate,
        "avg_ruff_score": avg_ruff_score,
        "avg_mypy_score": avg_mypy_score,
        "avg_overall_score": avg_overall_score,
        "syntax_error_count": syntax_error_count,
        "syntax_error_rate": syntax_error_rate,
        "perfect_solutions": perfect_solutions,
        "perfect_solution_rate": perfect_solution_rate,
    }
    
    output_data = {
        "model_path": model_name,
        "aggregated_metrics": aggregated_metrics,
        "detailed_results": results,
    }
    
    output_file = output_dir / f"{model_name.replace('/', '_')}_evaluation.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    return output_file


def print_metrics(metrics: EvaluationMetrics):
    """Print evaluation metrics."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Test pass rate: {metrics.test_pass_rate:.2%} ({metrics.test_passed}/{metrics.test_total})")
    print(f"Ruff score: {metrics.ruff_score:.4f}")
    print(f"Mypy score: {metrics.mypy_score:.4f}")
    print(f"Overall score: {metrics.overall_score:.4f}")
    print(f"Syntax error: {metrics.syntax_error}")
    if metrics.stderr:
        print(f"\nStderr output:\n{metrics.stderr[:500]}...")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Manually input and evaluate model completions for ClassEval"
    )
    parser.add_argument(
        "--class_name",
        type=str,
        default="Thermostat",
        help="Name of the class to evaluate (default: Thermostat)",
    )
    parser.add_argument(
        "--completion",
        type=str,
        default=None,
        help="The model completion (code). If not provided, will read from stdin.",
    )
    parser.add_argument(
        "--completion_file",
        type=str,
        default=None,
        help="Path to file containing the completion code",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model (used for output filename)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./evaluation_results",
        help="Directory to save results",
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
    
    # Get completion code
    if args.completion:
        completion = args.completion
    elif args.completion_file:
        with open(args.completion_file, "r", encoding="utf-8") as f:
            completion = f.read()
    else:
        print(f"\nEnter the completion for class '{args.class_name}'.")
        print("Paste the code and press Ctrl+D (Linux/Mac) or Ctrl+Z (Windows) when done:")
        print("-" * 60)
        import sys
        completion = sys.stdin.read()
    
    print(f"\nEvaluating completion for class '{args.class_name}'...")
    
    # Evaluate
    result, metrics = evaluate_manual_completion(
        class_name=args.class_name,
        completion=completion,
        tests_weight=args.tests_weight,
        ruff_weight=args.ruff_weight,
        mypy_weight=args.mypy_weight,
    )
    
    # Print results
    print_metrics(metrics)
    
    # Save results
    output_dir = Path(args.output_dir)
    save_evaluation_result(
        model_name=args.model_name,
        results=[result],
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
