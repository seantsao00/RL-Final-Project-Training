import json
from pathlib import Path
from dataclasses import asdict

from tqdm import tqdm

from src.env_classeval import (
    run_unittest,
    evaluate_ruff,
    evaluate_mypy,
    ClassEvalExecutionResult,
)


def run_evaluation_and_save(
    test_files: list[Path],
    composed_files: list[Path],
    summary_file: Path,
    ruff_config: dict,
) -> list[dict[str, any]]:
    results_summary: dict[dict[str, any]] = {}
    composed_map: dict[str, Path] = {}
    for cf in composed_files:
        composed_map[cf.stem] = cf

    for full_test_file in test_files:
        class_name = full_test_file.stem.replace("full_test_", "")
        result: ClassEvalExecutionResult = run_unittest(full_test_file.read_text())

        class_file = composed_map.get(
            class_name, full_test_file.parent / f"{class_name}.py"
        )

        class_code = class_file.read_text()
        ruff_report = evaluate_ruff(
            class_code, ruff_config["select"], ruff_config["ignore"]
        )
        mypy_report = evaluate_mypy(class_code)

        test_status: dict[str, any] = {
            "class_name": class_name,
            "unittest_result": asdict(result),
            "ruff": asdict(ruff_report),
            "mypy": asdict(mypy_report),
        }
        if result.n_total > 0:
            test_status["passrate"] = result.n_passed / result.n_total
        else:
            test_status["passrate"] = 0.0

        results_summary[class_name] = test_status

    passed_testcases = sum(
        r["unittest_result"]["n_passed"] for r in results_summary.values()
    )
    total_testcases = sum(
        r["unittest_result"]["n_total"] for r in results_summary.values()
    )
    testcase_passrate = (
        (passed_testcases / total_testcases) if total_testcases > 0 else 0.0
    )

    total_class = len(results_summary)
    passed_classes = sum(
        1
        for r in results_summary.values()
        if (
            r["unittest_result"]["n_passed"] == r["unittest_result"]["n_total"]
            and r["unittest_result"]["n_total"] > 0
        )
    )
    class_passrate = (passed_classes / total_class) if total_class > 0 else 0.0

    total_syntax_errors = sum(
        1 for r in results_summary.values() if r["unittest_result"]["syntax_error"]
    )

    total_ruff_issues = sum(r["ruff"]["n_issues"] for r in results_summary.values())
    total_mypy_errors = sum(r["mypy"]["n_errors"] for r in results_summary.values())

    print(
        f"testcase passrate: {testcase_passrate} ({passed_testcases}/{total_testcases})"
    )
    print(f"total syntax errors: {total_syntax_errors}")
    print(f"class passrate: {class_passrate} ({passed_classes}/{total_class})")
    print(f"total ruff issues: {total_ruff_issues}")
    print(f"total mypy errors: {total_mypy_errors}")

    summary_payload = {
        "overall": {
            "passed_testcases": passed_testcases,
            "total_testcases": total_testcases,
            "testcase_passrate": testcase_passrate,
            "passed_class": passed_classes,
            "total_syntax_errors": total_syntax_errors,
            "total_class": total_class,
            "class_passrate": class_passrate,
            "total_ruff_issues": total_ruff_issues,
            "total_mypy_errors": total_mypy_errors,
        },
        "results": results_summary,
    }

    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(summary_payload, f, ensure_ascii=False, indent=2)

    print(
        f"Saved test summary to {summary_file} | Passrate: {passed_classes}/{total_class} ({class_passrate:.2f}%)"
    )

    return results_summary


def discover_eval_artifacts(eval_output_dir: Path) -> tuple[list[Path], list[Path]]:
    """Discover both full test files and composed class files.

    Structure expected from generate.py:
      - Tests: <eval_output_dir>/composed_code/<ClassName>/full_test_<ClassName>.py
      - Classes: <eval_output_dir>/composed_code/<ClassName>/<ClassName>.py
    Returns (test_files, composed_files).
    """
    code_dir = eval_output_dir / "composed_code"
    if not code_dir.exists():
        raise FileNotFoundError(f"Composed code directory not found: {code_dir}")

    test_files: list[Path] = []
    composed_files: list[Path] = []

    for cls_dir in code_dir.iterdir():
        if not cls_dir.is_dir():
            continue
        for f in cls_dir.glob("full_test_*.py"):
            test_files.append(f)
        for f in cls_dir.glob("*.py"):
            if f.name.startswith("full_test_"):
                continue
            composed_files.append(f)

    if not test_files:
        raise FileNotFoundError(
            f"No test files found under {code_dir}. Expected full_test_*.py per class."
        )
    if not composed_files:
        raise FileNotFoundError(
            f"No composed class files found under {code_dir}. Expected <ClassName>.py per class."
        )

    return sorted(test_files), sorted(composed_files)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run ClassEval unit tests from generated results"
    )
    parser.add_argument(
        "--eval_output_dir",
        type=str,
        required=True,
        help=(
            "Directory containing generation results (e.g., classeval_eval/results/eval_result). "
            "Should include a 'composed_code' subdirectory with full_test_*.py files."
        ),
    )
    parser.add_argument(
        "--summary_file",
        type=str,
        default="results_summary.json",
        help=("Optional path to write the summary JSON"),
    )

    args = parser.parse_args()
    eval_output_dir = Path(args.eval_output_dir)
    test_files, composed_files = discover_eval_artifacts(eval_output_dir)

    summary_file = (
        Path(args.summary_file)
        if args.summary_file is not None
        else eval_output_dir / "results_summary.json"
    )

    run_evaluation_and_save(
        test_files=test_files,
        composed_files=composed_files,
        summary_file=summary_file,
        ruff_config={
            "select": ["F", "E", "W", "C90", "N", "UP", "B", "A", "C4", "RET", "SIM", "ARG"],
            "ignore": ["E501", "E741", "W292"],
        }
    )
