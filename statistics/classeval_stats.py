import argparse
import json
import os
import re
import ast
from typing import Tuple

try:
    from datasets import load_dataset  # type: ignore
except Exception:
    load_dataset = None


def count_lines(text: str, include_empty: bool = True) -> int:
    """Count lines in a string."""
    if not text:
        return 0
    lines = text.splitlines()
    if include_empty:
        return len(lines)
    return sum(1 for l in lines if l.strip())


def count_words(text: str) -> int:
    """Count words by whitespace."""
    if not text:
        return 0
    return len(re.findall(r"\S+", text))


def compute_split_stats(
    ds_split, include_empty: bool, max_rows: int | None
) -> Tuple[float, float, int, int]:
    """
    Returns:
    - avg_lines_per_solution
    - avg_words_per_question
    - total_solutions
    - total_questions
    """
    if max_rows is not None:
        n = min(max_rows, len(ds_split))
        ds_split = ds_split.select(range(n))

    total_solution_lines = 0
    total_solutions = 0
    total_question_lines = 0
    total_questions = 0

    for ex in ds_split:
        solution = ex["solution_code"]
        total_solution_lines += count_lines(
            solution, include_empty=include_empty
        )
        total_solutions += 1

        question = ex["skeleton"]
        total_question_lines += count_lines(question, include_empty=include_empty)
        total_questions += 1

    avg_lines = total_solution_lines / total_solutions if total_solutions else 0.0
    avg_q_lines = total_question_lines / total_questions if total_questions else 0.0
    return avg_lines, avg_q_lines, total_solutions, total_questions



def main():
    parser = argparse.ArgumentParser(
        description="Compute overall stats for ClassEval dataset (across all splits)."
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
        help="Include empty lines when counting solution lines.",
    )
    parser.add_argument(
        "--save_dataset",
        type=str,
        default=None,
        help=(
            "Path to store dataset as JSON/JSONL or directory with per-split files."
        ),
    )
    args = parser.parse_args()

    ds = load_dataset("FudanSELab/ClassEval")

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

        if ext == ".json":
            combined = {sp: list(cap_iter(ds[sp])) for sp in splits}
            with open(target, "w", encoding="utf-8") as f:
                json.dump(combined, f, ensure_ascii=False, indent=2)
            print(f"Saved combined JSON to {target}")

        elif ext == ".jsonl":
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

    # Aggregate across all available splits (e.g., 'test' only)
    total_solution_lines_sum = 0.0
    total_question_lines_sum = 0.0
    total_solutions = 0
    total_questions = 0

    for split in ds.keys():
        avg_lines, avg_q_lines, split_solutions, split_questions = compute_split_stats(
            ds[split], args.include_empty, args.max_rows
        )
        total_solution_lines_sum += avg_lines * split_solutions
        total_question_lines_sum += avg_q_lines * split_questions
        total_solutions += split_solutions
        total_questions += split_questions

    all_avg_lines = (total_solution_lines_sum / total_solutions) if total_solutions else 0.0
    all_avg_q_lines = (total_question_lines_sum / total_questions) if total_questions else 0.0

    print("Stats (average per item, all splits combined):")
    print(f"- Solution lines: all={all_avg_lines}")
    print(f"- Question lines: all={all_avg_q_lines}")

if __name__ == "__main__":
    main()
