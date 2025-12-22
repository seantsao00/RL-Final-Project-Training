import argparse
import json
import os
import math
import re
from typing import Tuple

try:
    from datasets import load_dataset  # type: ignore
except Exception:
    load_dataset = None


def count_lines(text: str, include_empty: bool = True) -> int:
    """Count lines in a string.

    If `include_empty` is False, only non-empty lines are counted.
    """
    if not text:
        return 0
    lines = text.splitlines()
    if include_empty:
        return len(lines)
    return sum(1 for l in lines if l.strip())


def count_words(text: str) -> int:
    """Count words by splitting on whitespace-like sequences."""
    if not text:
        return 0
    return len(re.findall(r"\S+", text))


def compute_split_stats(ds_split, include_empty: bool, max_rows: int | None) -> Tuple[float, float, int, int]:
    """Compute stats for a single split.

    Returns:
    - avg_lines_per_solution
    - avg_words_per_prompt
    - total_solutions
    - total_prompts
    """
    if max_rows is not None:
        n = min(max_rows, len(ds_split))
        ds_split = ds_split.select(range(n))

    total_solution_lines = 0
    total_solutions = 0
    total_prompt_words = 0
    total_prompts = 0

    for ex in ds_split:
        solution = ex.get("canonical_solution", "")
        total_solution_lines += count_lines(solution, include_empty=include_empty)
        total_solutions += 1

        prompt = ex.get("prompt", "")
        total_prompt_words += count_words(prompt)
        total_prompts += 1

    avg_lines = (total_solution_lines / total_solutions) if total_solutions else 0.0
    avg_words = (total_prompt_words / total_prompts) if total_prompts else 0.0
    return avg_lines, avg_words, total_solutions, total_prompts


def format_float(x: float) -> str:
    if math.isnan(x) or math.isinf(x):
        return "0.0"
    return f"{x:.2f}"


def main():
    parser = argparse.ArgumentParser(description="Compute overall stats for HumanEval (across all splits).")
    parser.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="Limit rows per split for faster runs (optional).",
    )
    parser.add_argument(
        "--include_empty",
        action="store_true",
        help="Include empty lines when counting solution lines.",
    )
    parser.add_argument(
        "--save_dataset",
        type=str,
        default=None,
        help=(
            "Path to store dataset as JSON/JSONL or directory with per-split files. "
            "If a directory, writes per-split files (e.g., test.json). "
            "If a file ending in .json, writes a combined object with splits. "
            "If a file ending in .jsonl, writes JSON Lines; for multiple splits, each line includes a 'split' field."
        ),
    )
    args = parser.parse_args()

    if load_dataset is None:
        raise RuntimeError("The 'datasets' library is not available. Install with 'pip install datasets'.")

    ds = load_dataset("openai_humaneval")

    # Optionally persist the loaded dataset as JSON
    if args.save_dataset:
        target = args.save_dataset
        _, ext = os.path.splitext(target)
        splits = list(ds.keys())

        def cap_iter(dsplit):
            if args.max_rows is None:
                for ex in dsplit:
                    yield dict(ex)
            else:
                n = min(args.max_rows, len(dsplit))
                for ex in dsplit.select(range(n)):
                    yield dict(ex)

        if ext.lower() == ".json":
            combined = {sp: list(cap_iter(ds[sp])) for sp in splits}
            with open(target, "w", encoding="utf-8") as f:
                json.dump(combined, f, ensure_ascii=False, indent=2)
            print(f"Saved combined JSON to {target}")
        elif ext.lower() == ".jsonl":
            with open(target, "w", encoding="utf-8") as f:
                for sp in splits:
                    for ex in cap_iter(ds[sp]):
                        ex["split"] = sp
                        f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            print(f"Saved JSONL to {target}")
        else:
            os.makedirs(target, exist_ok=True)
            for sp in splits:
                out = os.path.join(target, f"{sp}.json")
                with open(out, "w", encoding="utf-8") as f:
                    json.dump(list(cap_iter(ds[sp])), f, ensure_ascii=False, indent=2)
                print(f"Saved {sp} split to {out}")

    # Aggregate across all splits (HumanEval typically has 'test' only)
    total_solution_lines_sum = 0.0
    total_prompt_words_sum = 0.0
    total_solutions = 0
    total_prompts = 0

    for split in ds.keys():
        avg_lines, avg_words, split_solutions, split_prompts = compute_split_stats(
            ds[split], args.include_empty, args.max_rows
        )
        total_solution_lines_sum += avg_lines * split_solutions
        total_prompt_words_sum += avg_words * split_prompts
        total_solutions += split_solutions
        total_prompts += split_prompts

    all_avg_lines = (total_solution_lines_sum / total_solutions) if total_solutions else 0.0
    all_avg_words = (total_prompt_words_sum / total_prompts) if total_prompts else 0.0

    print("Stats (average per item, all splits combined):")
    print(f"- Solution lines: all={format_float(all_avg_lines)}")
    print(f"- Prompt words: all={format_float(all_avg_words)}")


if __name__ == "__main__":
    main()
