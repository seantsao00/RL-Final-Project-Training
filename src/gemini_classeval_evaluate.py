"""
Evaluation script for Gemini model on ClassEval holistic dataset.
"""
import argparse
import json
import os
from pathlib import Path
from dataclasses import asdict
from typing import Any

import google.genai as genai
from google.genai import types
from tqdm import tqdm
from datasets import Dataset

from src.holisitc_classeval.data import load_classeval_holistic_dataset_prompt_only
from src.classeval_holistic_evaluate import (
    evaluate_single_example,
    AggregatedMetrics,
    EvaluationMetrics,
)


def generate_completion_gemini(
    client: genai.Client,
    model_id: str,
    prompt: list[dict[str, str]],
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
) -> str:
    """Generate a completion using Gemini model with google.genai API."""
    # Convert prompt format to google.genai format
    # google.genai uses system instruction and contents separately
    system_instruction = None
    contents = []
    
    for msg in prompt:
        if msg["role"] == "system":
            system_instruction = msg["content"]
        else:
            contents.append(types.Content(
                role="user" if msg["role"] == "user" else "model",
                parts=[types.Part(text=msg["content"])]
            ))
    
    # Generate response
    response = client.models.generate_content(
        model=model_id,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
            max_output_tokens=max_new_tokens,
        )
    )
    
    # Generate response
    response = client.models.generate_content(
        model=model_id,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
            max_output_tokens=max_new_tokens,
        )
    )
    
    return response.text


def evaluate_model_gemini(
    model_name: str,
    dataset: Dataset,
    output_file: Path | None = None,
    max_examples: int | None = None,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    tests_weight: float = 1.0,
    ruff_weight: float = 0.5,
    mypy_weight: float = 1.5,
    api_key: str | None = None,
) -> tuple[AggregatedMetrics, list[dict[str, Any]]]:
    """
    Evaluate Gemini model on the dataset.
    
    Returns:
        - Aggregated metrics
        - List of detailed results for each example
    """
    # Configure Gemini API with google.genai Client
    if api_key:
        client = genai.Client(api_key=api_key)
    elif os.getenv("GEMINI_API_KEY"):
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    else:
        raise ValueError("API key must be provided via --api_key or GEMINI_API_KEY environment variable")
    
    print(f"Using Gemini model: {model_name}")
    
    # Limit dataset size if specified
    if max_examples is not None:
        dataset = dataset.select(range(min(len(dataset), max_examples)))
    
    print(f"Evaluating {len(dataset)} examples...")
    
    all_metrics = []
    detailed_results = []
    
    for idx, example in enumerate(tqdm(dataset, desc="Evaluating")):
        prompt = example["prompt"]
        tests = example["tests"]
        
        try:
            # Generate completion
            completion = generate_completion_gemini(
                client=client,
                model_id=model_name,
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
            
        except Exception as e:
            print(f"\nError on example {idx}: {e}")
            # Add failed result
            metrics = EvaluationMetrics(
                test_passed=0,
                test_total=0,
                test_pass_rate=0.0,
                syntax_error=True,
                ruff_score=0.0,
                mypy_score=0.0,
                overall_score=0.0,
                stderr=str(e),
            )
            all_metrics.append(metrics)
            detailed_results.append({
                "example_id": idx,
                "prompt": prompt,
                "completion": f"ERROR: {e}",
                "tests": tests,
                "metrics": asdict(metrics),
            })
        
        # Print progress every 5 examples
        if (idx + 1) % 5 == 0:
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
            "aggregated_metrics": asdict(aggregated),
            "detailed_results": detailed_results,
        }
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_file}")
    
    return aggregated, detailed_results


def print_metrics(metrics: AggregatedMetrics):
    """Pretty print the aggregated metrics."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
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
        description="Evaluate Gemini model on ClassEval holistic dataset"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gemini-1.5-flash",
        help="Gemini model name (default: gemini-1.5-flash)",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="Gemini API key (or set GEMINI_API_KEY environment variable)",
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
    output_file = Path(args.output_dir) / f"{args.model_name.replace('/', '_')}_evaluation.json"
    
    # Run evaluation
    metrics, detailed_results = evaluate_model_gemini(
        model_name=args.model_name,
        dataset=test_dataset,
        output_file=output_file,
        max_examples=args.max_examples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        tests_weight=args.tests_weight,
        ruff_weight=args.ruff_weight,
        mypy_weight=args.mypy_weight,
        api_key=args.api_key,
    )
    
    # Print results
    print_metrics(metrics)


if __name__ == "__main__":
    main()
