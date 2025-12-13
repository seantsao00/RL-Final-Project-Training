#!/usr/bin/env python3
import json
import subprocess
import tempfile
from pathlib import Path

RUFF_SELECT = [
    "F", 
    "E",
    "W",
    "C90",
    "N",
    "UP",
    "B",
    "A",
    "C4",
    "RET",
    "SIM",
    "ARG",
]

RUFF_IGNORE = [
    "E501",
    "E741",
    "W292",
]

SYNTAX_ERROR_PENALTY = -1.0

def check_syntax_error(code: str) -> bool:
    try:
        compile(code, "<string>", "exec")
        return False
    except SyntaxError:
        return True


def evaluate_ruff_standalone(code: str) -> tuple[bool, int, list[str]]:
    if check_syntax_error(code):
        return True, 0, ["Syntax error"]
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "code.py"
        tmp_path.write_text(code)
        try:
            result = subprocess.run(
                [
                    "ruff",
                    "check",
                    "--select=" + ",".join(RUFF_SELECT),
                    "--ignore=" + ",".join(RUFF_IGNORE),
                    "--output-format=json",
                    str(tmp_path),
                ],
                capture_output=True,
                text=True,
                timeout=10.0,
            )
            if result.stdout:
                issues = json.loads(result.stdout)
                n_issues = len(issues) if isinstance(issues, list) else 0
                messages = [issue.get("message", "") for issue in issues] if isinstance(issues, list) else []
            else:
                n_issues = 0
                messages = []
                
            return False, n_issues, messages
            
        except Exception as e:
            print(f"Ruff error: {e}")
            return False, 0, []


def evaluate_mypy_standalone(code: str) -> tuple[bool, int, list[str]]:
    if check_syntax_error(code):
        return True, 0, ["Syntax error"]
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "code.py"
        tmp_path.write_text(code)
        try:
            result = subprocess.run(
                [
                    "mypy",
                    "--strict",
                    "--no-color-output",
                    "--no-error-summary",
                    str(tmp_path),
                ],
                capture_output=True,
                text=True,
                timeout=10.0,
            )
            error_lines = [
                line for line in result.stdout.splitlines() if ": error:" in line
            ]
            n_errors = len(error_lines)
            messages = error_lines
            
            return False, n_errors, messages
            
        except Exception as e:
            print(f"Mypy error: {e}")
            return False, 0, []


def calculate_ruff_reward(code: str) -> float:
    syntax_error, n_issues, _ = evaluate_ruff_standalone(code)
    if syntax_error:
        return SYNTAX_ERROR_PENALTY
    return 1.0 / (1.0 + n_issues)


def calculate_mypy_reward(code: str) -> float:
    syntax_error, n_errors, _ = evaluate_mypy_standalone(code)
    if syntax_error:
        return SYNTAX_ERROR_PENALTY
    return 1.0 / (1.0 + n_errors)


def analyze_file(file_path: str, model_name: str):
    completions = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                completions.append(json.loads(line))
    
    ruff_rewards = []
    mypy_rewards = []
    syntax_errors = 0
    
    for i, completion in enumerate(completions):
        task_id = completion["task_id"]
        code = completion["completion"]
        
        ruff_reward = calculate_ruff_reward(code)
        mypy_reward = calculate_mypy_reward(code)
        ruff_rewards.append(ruff_reward)
        mypy_rewards.append(mypy_reward)
        
        if ruff_reward == SYNTAX_ERROR_PENALTY or mypy_reward == SYNTAX_ERROR_PENALTY:
            syntax_errors += 1
    
    avg_ruff = sum(ruff_rewards) / len(ruff_rewards)
    avg_mypy = sum(mypy_rewards) / len(mypy_rewards)
    ruff_no_syntax = [r for r in ruff_rewards if r != SYNTAX_ERROR_PENALTY]
    mypy_no_syntax = [r for r in mypy_rewards if r != SYNTAX_ERROR_PENALTY]
    avg_ruff_no_syntax = sum(ruff_no_syntax) / len(ruff_no_syntax) if ruff_no_syntax else 0
    avg_mypy_no_syntax = sum(mypy_no_syntax) / len(mypy_no_syntax) if mypy_no_syntax else 0
    perfect_ruff = sum(1 for r in ruff_rewards if r == 1.0)
    perfect_mypy = sum(1 for r in mypy_rewards if r == 1.0)
    
    print(f"{'='*80}")
    print(f"STATISTICS - {model_name}")
    print(f"{'='*80}")
    print(f"\nSyntax Errors:")
    print(f"  Count: {syntax_errors}")
    print(f"  Percentage: {syntax_errors / len(completions) * 100:.2f}%")
    
    print(f"\nRuff Rewards:")
    print(f"  Average (all): {avg_ruff:.4f}")
    print(f"  Average (no syntax errors): {avg_ruff_no_syntax:.4f}")
    print(f"  Perfect scores (1.0): {perfect_ruff} ({perfect_ruff / len(completions) * 100:.2f}%)")
    print(f"  Min: {min(ruff_rewards):.4f}")
    print(f"  Max: {max(ruff_rewards):.4f}")
    
    print(f"\nMypy Rewards:")
    print(f"  Average (all): {avg_mypy:.4f}")
    print(f"  Average (no syntax errors): {avg_mypy_no_syntax:.4f}")
    print(f"  Perfect scores (1.0): {perfect_mypy} ({perfect_mypy / len(completions) * 100:.2f}%)")
    print(f"  Min: {min(mypy_rewards):.4f}")
    print(f"  Max: {max(mypy_rewards):.4f}")
    
    print(f"\nCombined Reward (0.5*ruff + 0.5*mypy):")
    combined_rewards = [(r + m) / 2 for r, m in zip(ruff_rewards, mypy_rewards)]
    avg_combined = sum(combined_rewards) / len(combined_rewards)
    print(f"  Average: {avg_combined:.4f}")
    
    return {
        "model": model_name,
        "file": file_path,
        "total": len(completions),
        "syntax_errors": syntax_errors,
        "ruff_avg": avg_ruff,
        "ruff_avg_no_syntax": avg_ruff_no_syntax,
        "ruff_perfect": perfect_ruff,
        "mypy_avg": avg_mypy,
        "mypy_avg_no_syntax": avg_mypy_no_syntax,
        "mypy_perfect": perfect_mypy,
        "combined_avg": avg_combined,
    }


def main():
    print("\n" + "="*80)
    print("HUMANEVAL REWARD ANALYSIS")
    print("="*80)
    
    # Analyze both files
    results = []
    
    # Analyze base model
    results.append(analyze_file(
        "base_results.jsonl",
        "Qwen2.5-Coder-7B-Instruct (Base)"
    ))
    
    # Analyze GRPO trained model
    results.append(analyze_file(
        "grpo7b.jsonl",
        "GRPO-7B (Fine-tuned)"
    ))
    
    # Comparison
    print(f"\n{'='*80}")
    print("COMPARISON")
    print(f"{'='*80}\n")
    
    base = results[0]
    grpo = results[1]
    
    print(f"{'Metric':<40} {'Base':<15} {'GRPO':<15} {'Change':<15}")
    print("-" * 80)
    
    metrics = [
        ("Syntax Errors", "syntax_errors", False),
        ("Ruff Reward (avg)", "ruff_avg", True),
        ("Ruff Reward (no syntax)", "ruff_avg_no_syntax", True),
        ("Ruff Perfect Scores", "ruff_perfect", False),
        ("Mypy Reward (avg)", "mypy_avg", True),
        ("Mypy Reward (no syntax)", "mypy_avg_no_syntax", True),
        ("Mypy Perfect Scores", "mypy_perfect", False),
        ("Combined Reward", "combined_avg", True),
    ]
    
    for metric_name, key, is_float in metrics:
        base_val = base[key]
        grpo_val = grpo[key]
        change = grpo_val - base_val
        
        if is_float:
            base_str = f"{base_val:.4f}"
            grpo_str = f"{grpo_val:.4f}"
            change_str = f"{change:+.4f}"
        else:
            base_str = str(base_val)
            grpo_str = str(grpo_val)
            change_str = f"{change:+d}"
        
        print(f"{metric_name:<40} {base_str:<15} {grpo_str:<15} {change_str:<15}")
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

