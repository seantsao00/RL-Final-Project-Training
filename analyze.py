#!/usr/bin/env python3
"""
Calculate rewards (syntax error, mypy, and ruff) for APPS dataset completions.

This script evaluates code completions from APPS dataset and calculates:
- Syntax error detection
- Ruff reward (based on code quality issues)
- Mypy reward (based on type checking errors)

Usage:
    python calculate_rewards_apps.py --input apps_eval/7bbaseline.jsonl
"""
import argparse
import json
import statistics
import sys
import os
import re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.env import evaluate_ruff, evaluate_mypy

SYNTAX_ERROR_PENALTY = -1.0


def extract_code_from_completion(completion: str) -> str:
    """Extract Python code from completion, handling markdown code blocks."""
    match = re.search(r"```python(.*?)```", completion, re.DOTALL)
    if match:
        code = match.group(1).strip()
    else:
        match = re.search(r"```(.*?)```", completion, re.DOTALL)
        if match:
            code = match.group(1).strip()
        else:
            code = completion.strip()
    
    return code


def check_syntax_error(code: str) -> bool:
    """Check if code has syntax errors."""
    try:
        compile(code, "<string>", "exec")
        return False
    except SyntaxError:
        return True
    except Exception:
        # Other compilation errors (not syntax errors)
        return False


def process_file(file_path: str):
    """Process APPS completions file and calculate rewards."""
    completions = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                completions.append(json.loads(line))
    
    syntax_errors = 0
    ruff_rewards = []
    mypy_rewards = []
    ruff_issues_list = []
    mypy_errors_list = []
    ruff_syntax_errors = 0
    mypy_syntax_errors = 0
    
    print(f"Processing {len(completions)} completions...")
    
    for idx, completion_entry in enumerate(completions):
        if idx % 100 == 0 and idx > 0:
            print(f"  Processed {idx}/{len(completions)} completions...")
        
        completion_code = completion_entry.get("completion", "")
        
        # Extract code from completion (handles markdown code blocks)
        code = extract_code_from_completion(completion_code)
        
        # Check for syntax errors
        has_syntax_error = check_syntax_error(code)
        if has_syntax_error:
            syntax_errors += 1
        
        # Evaluate ruff
        ruff_result = evaluate_ruff(
            code,
            select=["F", "E", "W", "C90", "N", "UP", "B", "A", "C4", "RET", "SIM", "ARG"],
            ignore=["E501", "E741", "W292"]
        )
        
        # Calculate ruff reward
        if ruff_result.syntax_error:
            ruff_reward = SYNTAX_ERROR_PENALTY
            ruff_syntax_errors += 1
        else:
            ruff_reward = 1.0 / (1.0 + ruff_result.n_issues)
        
        ruff_rewards.append(ruff_reward)
        ruff_issues_list.append(ruff_result.n_issues)
        
        # Evaluate mypy
        mypy_result = evaluate_mypy(code)
        
        # Calculate mypy reward
        if mypy_result.syntax_error:
            mypy_reward = SYNTAX_ERROR_PENALTY
            mypy_syntax_errors += 1
        else:
            mypy_reward = 1.0 / (1.0 + mypy_result.n_errors)
        
        mypy_rewards.append(mypy_reward)
        mypy_errors_list.append(mypy_result.n_errors)
    
    return {
        "syntax_errors": syntax_errors,
        "ruff_rewards": ruff_rewards,
        "mypy_rewards": mypy_rewards,
        "ruff_issues": ruff_issues_list,
        "mypy_errors": mypy_errors_list,
        "ruff_syntax_errors": ruff_syntax_errors,
        "mypy_syntax_errors": mypy_syntax_errors,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Calculate rewards (syntax error, mypy, ruff) for APPS dataset completions"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input JSONL file path (e.g., apps_eval/7bbaseline.jsonl)"
    )
    parser.add_argument(
        "--output",
        help="Optional output JSON file to save detailed results"
    )
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        sys.exit(1)
    
    print(f"Evaluating rewards for: {args.input}")
    print("=" * 80)
    
    results = process_file(args.input)
    
    # Calculate statistics
    total_completions = len(results["ruff_rewards"])
    syntax_errors = results["syntax_errors"]
    ruff_syntax_errors = results["ruff_syntax_errors"]
    mypy_syntax_errors = results["mypy_syntax_errors"]
    
    avg_ruff_issues = statistics.mean(results["ruff_issues"])
    avg_mypy_errors = statistics.mean(results["mypy_errors"])
    
    avg_ruff_reward = statistics.mean(results["ruff_rewards"])
    avg_mypy_reward = statistics.mean(results["mypy_rewards"])
    
    # Calculate reward based on average issues/errors
    ruff_reward_from_avg = 1.0 / (1.0 + avg_ruff_issues)
    mypy_reward_from_avg = 1.0 / (1.0 + avg_mypy_errors)
    
    # Print results
    print("\n" + "=" * 80)
    print("REWARD EVALUATION RESULTS")
    print("=" * 80)
    print(f"Total completions evaluated: {total_completions}")
    print()
    
    print("Syntax Errors:")
    print(f"  Python syntax errors (compile check): {syntax_errors} ({syntax_errors/total_completions*100:.2f}%)")
    print(f"  Ruff detected syntax errors: {ruff_syntax_errors} ({ruff_syntax_errors/total_completions*100:.2f}%)")
    print(f"  Mypy detected syntax errors: {mypy_syntax_errors} ({mypy_syntax_errors/total_completions*100:.2f}%)")
    print(f"  Total syntax errors (any source): {syntax_errors + ruff_syntax_errors + mypy_syntax_errors}")
    print()
    
    print("Ruff Rewards:")
    print(f"  Average ruff issues per completion: {avg_ruff_issues:.2f}")
    print(f"  Average ruff reward: {avg_ruff_reward:.4f}")
    print(f"  Ruff reward (from avg issues): {ruff_reward_from_avg:.4f}")
    print()
    
    print("Mypy Rewards:")
    print(f"  Average mypy errors per completion: {avg_mypy_errors:.2f}")
    print(f"  Average mypy reward: {avg_mypy_reward:.4f}")
    print(f"  Mypy reward (from avg errors): {mypy_reward_from_avg:.4f}")
    print()
    
    print("Summary:")
    print(f"  Syntax error rate: {syntax_errors/total_completions*100:.2f}%")
    print(f"  Average ruff reward: {avg_ruff_reward:.4f}")
    print(f"  Average mypy reward: {avg_mypy_reward:.4f}")
    print("=" * 80)
    
    # Save detailed results if output file specified
    if args.output:
        output_data = {
            "input_file": args.input,
            "total_completions": total_completions,
            "syntax_errors": {
                "python_compile": syntax_errors,
                "ruff_detected": ruff_syntax_errors,
                "mypy_detected": mypy_syntax_errors,
            },
            "ruff": {
                "average_issues": avg_ruff_issues,
                "average_reward": avg_ruff_reward,
                "reward_from_avg": ruff_reward_from_avg,
            },
            "mypy": {
                "average_errors": avg_mypy_errors,
                "average_reward": avg_mypy_reward,
                "reward_from_avg": mypy_reward_from_avg,
            },
            "per_completion": [
                {
                    "ruff_reward": r_reward,
                    "mypy_reward": m_reward,
                    "ruff_issues": r_issues,
                    "mypy_errors": m_errors,
                }
                for r_reward, m_reward, r_issues, m_errors in zip(
                    results["ruff_rewards"],
                    results["mypy_rewards"],
                    results["ruff_issues"],
                    results["mypy_errors"],
                )
            ],
        }
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    main()





