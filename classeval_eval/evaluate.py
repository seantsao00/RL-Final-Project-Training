import json
from pathlib import Path
from typing import Dict, Any

from tqdm import tqdm

from src.env_classeval import (
    add_timeout_to_unittest_code,
    run_unittest,
    ClassEvalExecutionResult,
)


def run_evaluation_and_save(
    composed_classes: dict[str, str],
    code_metadata: dict[str, Dict[str, Any]],
    code_dir: Path,
    summary_file: Path,
) -> list[Dict[str, Any]]:
    results_summary: list[Dict[str, Any]] = []

    for class_name in tqdm(
        list(composed_classes.keys()), desc="Running unit tests per class"
    ):
        unittest_code = code_metadata[class_name]["class_test_code"]
        unittest_code = add_timeout_to_unittest_code(unittest_code)
        full_test_code = f"""
{composed_classes[class_name]}

{unittest_code}
"""
        if not "unittest.main()" in full_test_code:
            full_test_code += """

if __name__ == "__main__":
    unittest.main()
"""

        full_test_file = code_dir / class_name / f"full_test_{class_name}.py"
        full_test_file.parent.mkdir(parents=True, exist_ok=True)
        with full_test_file.open("w", encoding="utf-8") as f:
            f.write(full_test_code)

        result: ClassEvalExecutionResult = run_unittest(full_test_file.read_text())

        test_status: Dict[str, Any] = {
            "class_name": class_name,
            "n_passed": result.n_passed,
            "n_total": result.n_total,
            "timed_out": result.timed_out,
            "runtime_error": result.runtime_error,
            "syntax_error": result.syntax_error,
            "stderr": result.stderr,
        }
        if result.n_total > 0:
            test_status["passrate"] = result.n_passed / result.n_total
        else:
            test_status["passrate"] = 0.0

        results_summary.append(test_status)

    passed_testcases = sum(r["n_passed"] for r in results_summary)
    total_testcases = sum(r["n_total"] for r in results_summary)
    testcase_passrate = (
        (passed_testcases / total_testcases) if total_testcases > 0 else 0.0
    )

    total_class = len(results_summary)
    passed_classes = sum(1 for r in results_summary if r["n_passed"] == r["n_total"])
    class_passrate = (passed_classes / total_class) if total_class > 0 else 0.0
    
    print(f"testcase passrate: {testcase_passrate} ({passed_testcases}/{total_testcases})")
    print(f"class passrate: {class_passrate} ({passed_classes}/{total_class})")

    summary_payload = {
        "overall": {
            "passed_testcases": passed_testcases,
            "total_testcases": total_testcases,
            "testcase_passrate": testcase_passrate,
            "passed_class": passed_classes,
            "total_class": total_class,
            "class_passrate": class_passrate,
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
