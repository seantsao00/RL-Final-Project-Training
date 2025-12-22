import argparse
import json
import os
import math
import re
import ast
from typing import Dict, Tuple

try:
    from datasets import load_dataset  # type: ignore
except Exception as e:  # pragma: no cover
    load_dataset = None  # Fallback for environments without datasets installed


def count_lines(text: str, include_empty: bool = True) -> int:
    """Count lines in a string.

    If `include_empty` is False, only non-empty lines are counted.
    """
    if text is None:
        return 0
    lines = text.splitlines()
    if include_empty:
        return len(lines)
    return sum(1 for l in lines if l.strip())


def count_words(text: str) -> int:
    """Count words in a question by splitting on whitespace-like sequences."""
    if not text:
        return 0
    return len(re.findall(r"\S+", text))


def compute_split_stats(
    ds_split, include_empty: bool, max_rows: int | None
) -> Tuple[float, float, int, int]:
    """Compute stats for a single split.

    Returns:
    - avg_lines_per_solution
    - avg_words_per_question
    - total_solutions
    - total_questions
    """
    if max_rows is not None:
        # Safely cap the number of rows processed
        n = min(max_rows, len(ds_split))
        ds_split = ds_split.select(range(n))

    total_solution_lines = 0
    total_solutions = 0
    total_question_words = 0
    total_questions = 0

    for ex in ds_split:
        solutions = ex.get("solutions")
        if not solutions:
            continue
        total_solution_lines += count_lines(
            ast.literal_eval(solutions)[0], include_empty=include_empty
        )
        total_solutions += 1

        q = ex.get("question", "")
        total_question_words += count_words(q)
        total_questions += 1

    avg_lines = (total_solution_lines / total_solutions) if total_solutions else 0.0
    avg_words = (total_question_words / total_questions) if total_questions else 0.0
    return avg_lines, avg_words, total_solutions, total_questions


def format_float(x: float) -> str:
    if math.isnan(x) or math.isinf(x):
        return "0.0"
    return f"{x:.2f}"


def main():
    parser = argparse.ArgumentParser(
        description="Compute stats for codeparrot/apps dataset."
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="Limit rows per split for faster runs (optional).",
    )
    parser.add_argument(
        "--include_empty",
        action="store_true",
        help="Include empty lines when counting solution lines (default off).",
    )
    parser.add_argument(
        "--save_dataset",
        type=str,
        default=None,
        help=(
            "Path to store the loaded dataset as JSON. "
            "If a directory, writes per-split files (e.g., train.json, test.json). "
            "If a file ending in .json, writes a combined object with splits. "
            "If a file ending in .jsonl, writes JSON Lines; for multiple splits, each line includes a 'split' field."
        ),
    )
    args = parser.parse_args()

    if load_dataset is None:
        raise RuntimeError(
            "The 'datasets' library is not available. Install with 'pip install datasets'."
        )

    ds = load_dataset("codeparrot/apps", trust_remote_code=True)

    # Optionally persist the loaded dataset as JSON
    if args.save_dataset:
        target = args.save_dataset
        base, ext = os.path.splitext(target)
        splits = list(ds.keys())

        def cap_iterable(dsplit):
            if args.max_rows is None:
                for ex in dsplit:
                    yield dict(ex)
            else:
                n = min(args.max_rows, len(dsplit))
                for ex in dsplit.select(range(n)):
                    yield dict(ex)

        if ext.lower() == ".json":
            combined = {}
            for sp in splits:
                combined[sp] = list(cap_iterable(ds[sp]))
            with open(target, "w", encoding="utf-8") as f:
                json.dump(combined, f, ensure_ascii=False, indent=2)
            print(f"Saved dataset JSON (combined) to: {target}")
        elif ext.lower() == ".jsonl":
            # Write all examples to a single JSONL; add 'split' for clarity
            with open(target, "w", encoding="utf-8") as f:
                for sp in splits:
                    for ex in cap_iterable(ds[sp]):
                        ex_with_split = {**ex, "split": sp}
                        f.write(json.dumps(ex_with_split, ensure_ascii=False) + "\n")
            print(f"Saved dataset JSONL to: {target}")
        else:
            # Treat as directory; write per-split JSON files
            os.makedirs(target, exist_ok=True)
            for sp in splits:
                out_path = os.path.join(target, f"{sp}.json")
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(
                        list(cap_iterable(ds[sp])), f, ensure_ascii=False, indent=2
                    )
                print(f"Saved split '{sp}' to: {out_path}")

    # Train stats
    train_avg_lines, train_avg_words, train_solutions, train_questions = (
        compute_split_stats(
            ds["train"], include_empty=args.include_empty, max_rows=args.max_rows
        )
    )

    # Test stats
    test_avg_lines, test_avg_words, test_solutions, test_questions = (
        compute_split_stats(
            ds["test"], include_empty=args.include_empty, max_rows=args.max_rows
        )
    )

    # All = weighted by counts across splits
    all_avg_lines = (
        (train_avg_lines * train_solutions + test_avg_lines * test_solutions)
        / (train_solutions + test_solutions)
        if (train_solutions + test_solutions) > 0
        else 0.0
    )
    all_avg_words = (
        (train_avg_words * train_questions + test_avg_words * test_questions)
        / (train_questions + test_questions)
        if (train_questions + test_questions) > 0
        else 0.0
    )

    print("Stats (average over items):")
    print(
        f"- Solution lines (per solution): train={format_float(train_avg_lines)}, "
        f"test={format_float(test_avg_lines)}, all={format_float(all_avg_lines)}"
    )
    print(
        f"- Question words (per question): train={format_float(train_avg_words)}, "
        f"test={format_float(test_avg_words)}, all={format_float(all_avg_words)}"
    )


if __name__ == "__main__":
    main()
