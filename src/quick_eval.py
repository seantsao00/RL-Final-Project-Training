#!/usr/bin/env python3
"""
Interactive script to quickly evaluate code snippets for ClassEval.
Just run it, paste your code, and it will generate the JSON evaluation.
"""
import json
from pathlib import Path
from dataclasses import asdict

from datasets import load_dataset
from src.holisitc_classeval.data import question_to_prompt
from src.classeval_holistic_evaluate import evaluate_single_example


def load_class_data(class_name: str):
    """Load the prompt and tests for a specific class."""
    dataset = load_dataset("FudanSELab/ClassEval", split="test", trust_remote_code=True)
    
    for row in dataset:
        if row["class_name"] == class_name:
            prompt = question_to_prompt(row["class_name"], row["skeleton"])
            return {
                "class_name": class_name,
                "prompt": prompt,
                "tests": row["test"],
            }
    
    raise ValueError(f"Class '{class_name}' not found in dataset")


def main():
    print("=" * 70)
    print("ClassEval Code Evaluator - Quick JSON Generator")
    print("=" * 70)
    
    # Get class name
    class_name = input("\nEnter class name (default: Thermostat): ").strip() or "Thermostat"
    
    # Get model name
    model_name = input("Enter model name (e.g., claude-3.5, gpt-4): ").strip()
    if not model_name:
        print("Error: Model name is required!")
        return
    
    print(f"\n✓ Evaluating for class: {class_name}")
    print(f"✓ Model name: {model_name}")
    
    # Load class data
    try:
        class_data = load_class_data(class_name)
    except ValueError as e:
        print(f"\nError: {e}")
        return
    
    # Get code from user
    print("\n" + "=" * 70)
    print("Paste your code below and press Ctrl+D (Linux/Mac) or Ctrl+Z+Enter (Windows):")
    print("=" * 70)
    

    
    completion = open("src/t.py", "r", encoding="utf-8").read()
    
    if not completion:
        print("\nError: No code provided!")
        return
    
    print("\n" + "=" * 70)
    print("Evaluating...")
    print("=" * 70)
    
    # Evaluate
    metrics = evaluate_single_example(
        completion=completion,
        tests=class_data["tests"],
        tests_weight=1.0,
        ruff_weight=0.5,
        mypy_weight=1.5,
    )
    
    # Build result
    result = {
        "example_id": 0,
        "class_name": class_name,
        "prompt": class_data["prompt"],
        "completion": completion,
        "tests": class_data["tests"],
        "metrics": asdict(metrics),
    }
    
    aggregated_metrics = {
        "num_examples": 1,
        "avg_test_pass_rate": metrics.test_pass_rate,
        "avg_ruff_score": metrics.ruff_score,
        "avg_mypy_score": metrics.mypy_score,
        "avg_overall_score": metrics.overall_score,
        "syntax_error_count": 1 if metrics.syntax_error else 0,
        "syntax_error_rate": 1.0 if metrics.syntax_error else 0.0,
        "perfect_solutions": 1 if metrics.test_pass_rate == 1.0 and not metrics.syntax_error else 0,
        "perfect_solution_rate": 1.0 if metrics.test_pass_rate == 1.0 and not metrics.syntax_error else 0.0,
    }
    
    output_data = {
        "model_path": model_name,
        "aggregated_metrics": aggregated_metrics,
        "detailed_results": [result],
    }
    
    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Test pass rate:  {metrics.test_pass_rate:.2%} ({metrics.test_passed}/{metrics.test_total})")
    print(f"Ruff score:      {metrics.ruff_score:.4f}")
    print(f"Mypy score:      {metrics.mypy_score:.4f}")
    print(f"Overall score:   {metrics.overall_score:.4f}")
    print(f"Syntax error:    {metrics.syntax_error}")
    print("=" * 70)
    
    # Save to file
    output_dir = Path("./evaluation_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{model_name.replace('/', '_')}_evaluation.json"
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ JSON saved to: {output_file}")
    print(f"\nYou can view it with: cat {output_file}")


if __name__ == "__main__":
    main()
