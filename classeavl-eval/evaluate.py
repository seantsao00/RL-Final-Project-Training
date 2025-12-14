import json
from pathlib import Path
from typing import Dict, Any
import subprocess

from tqdm import tqdm

from src.env_classeval import ClassEvalExecutionResult


def evaluate_classeval_candidate(full_test_code_path: Path, timeout_s: float = 10.0) -> ClassEvalExecutionResult:
    try:
        compile(full_test_code_path.read_text(), "<string>", "exec")
    except Exception as e:
        return ClassEvalExecutionResult(
            n_passed=0,
            n_total=0,
            timed_out=False,
            runtime_error=False,
            syntax_error=True,
            stderr=str(e),
        )

    result = None
    try:
        result = subprocess.run(
            ["python", "-m", "unittest", str(full_test_code_path)],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )

        stderr = result.stderr
        stdout = result.stdout

        n_total = 0
        n_passed = 0
        output = stderr + stdout

        if "Ran" in output:
            for line in output.split("\n"):
                if line.startswith("Ran "):
                    parts = line.split()
                    if len(parts) >= 2 and parts[1].isdigit():
                        n_total = int(parts[1])
                    break

        if result.returncode == 0:
            n_passed = n_total
        else:
            if "FAILED" in output:
                failures = 0
                errors = 0
                for line in output.split("\n"):
                    if "failures=" in line.lower():
                        try:
                            failures = int(line.split("failures=")[1].split(",")[0].split(")")[0].strip())
                        except Exception:
                            pass
                    if "errors=" in line.lower():
                        try:
                            errors = int(line.split("errors=")[1].split(",")[0].split(")")[0].strip())
                        except Exception:
                            pass
                n_passed = n_total - failures - errors

        return ClassEvalExecutionResult(
            n_passed=n_passed,
            n_total=n_total,
            timed_out=False,
            runtime_error=result.returncode != 0 and "Error" in output,
            syntax_error=False,
            stderr=stderr,
        )

    except subprocess.TimeoutExpired:
        stderr = result.stderr
        stdout = result.stdout
        print(stdout)
        return ClassEvalExecutionResult(
            n_passed=0,
            n_total=0,
            timed_out=True,
            runtime_error=False,
            syntax_error=False,
            stderr="Test execution timed out",
        )
    except Exception as e:
        return ClassEvalExecutionResult(
            n_passed=0,
            n_total=0,
            timed_out=False,
            runtime_error=True,
            syntax_error=False,
            stderr=str(e),
        )


def run_evaluation_and_save(
    composed_classes: dict[str, str],
    code_metadata: dict[str, Dict[str, Any]],
    code_dir: Path,
    summary_file: Path,
) -> list[Dict[str, Any]]:
    results_summary: list[Dict[str, Any]] = []

    for class_name in tqdm(list(composed_classes.keys()), desc="Running unit tests per class"):
        full_test_code = f"""
{composed_classes[class_name]}

{code_metadata[class_name]['class_test_code']}

if __name__ == "__main__":
    unittest.main()
"""
        full_test_file = code_dir / class_name / f"full_test_{class_name}.py"
        full_test_file.parent.mkdir(parents=True, exist_ok=True)
        with full_test_file.open("w", encoding="utf-8") as f:
            f.write(full_test_code)

        result = evaluate_classeval_candidate(full_test_code_path=full_test_file, timeout_s=10.0)

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

    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=2)

    return results_summary
