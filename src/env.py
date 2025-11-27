import json
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from .data.apps_plus import AppsSample


@dataclass
class ExecutionResult:
    n_passed: int
    n_total: int
    timed_out: bool
    runtime_error: bool
    stderr: str


@dataclass
class RuffResult:
    n_issues: int


@dataclass
class MypyResult:
    n_errors: int


def _run_single_test(
    candidate_path: Path, test_input: str, expected_output: str, timeout_s: float
) -> tuple[bool, bool, bool, str]:
    """Run a single test and return (passed, timed_out, runtime_error, stderr)."""
    try:
        result = subprocess.run(
            ["python", str(candidate_path)],
            input=test_input,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )

        actual_output = result.stdout.strip()
        stderr = result.stderr

        if result.returncode != 0:
            return False, False, True, stderr
        elif actual_output == expected_output:
            return True, False, False, stderr
        else:
            return False, False, False, stderr

    except subprocess.TimeoutExpired:
        return False, True, False, ""
    except Exception as e:
        return False, False, True, f"Exception: {str(e)}"


def run_tests_stdio(
    candidate_path: Path, tests: list[tuple[str, str]], timeout_s: float = 5.0
) -> ExecutionResult:
    n_total = len(tests)

    if n_total == 0:
        return ExecutionResult(
            n_passed=0, n_total=0, timed_out=False, runtime_error=False, stderr=""
        )

    n_passed = 0
    timed_out = False
    runtime_error = False
    stderr_output = ""

    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(_run_single_test, candidate_path, inp, exp, timeout_s): i
            for i, (inp, exp) in enumerate(tests)
        }

        for future in as_completed(futures):
            passed, timeout, error, stderr = future.result()
            if passed:
                n_passed += 1
            if timeout:
                timed_out = True
            if error:
                runtime_error = True
            if stderr:
                stderr_output += stderr

    return ExecutionResult(
        n_passed=n_passed,
        n_total=n_total,
        timed_out=timed_out,
        runtime_error=runtime_error,
        stderr=stderr_output,
    )





def run_ruff(candidate_path: Path) -> RuffResult:
    try:
        result = subprocess.run(
            ["ruff", "check", "--output-format", "json", str(candidate_path)],
            capture_output=True,
            text=True,
            timeout=10.0,
        )

        # Parse JSON output
        if result.stdout:
            issues = json.loads(result.stdout)
            n_issues = len(issues) if isinstance(issues, list) else 0
        else:
            n_issues = 0

    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception):
        # If ruff fails, assume no issues detected
        n_issues = 0

    return RuffResult(n_issues=n_issues)


def run_mypy(candidate_path: Path) -> MypyResult:
    try:
        result = subprocess.run(
            [
                "mypy",
                "--ignore-missing-imports",
                "--no-color-output",
                "--no-error-summary",
                str(candidate_path),
            ],
            capture_output=True,
            text=True,
            timeout=10.0,
        )

        # Count error lines in output
        # Mypy outputs errors like "file.py:line: error: message"
        error_lines = [
            line for line in result.stdout.splitlines() if ": error:" in line
        ]
        n_errors = len(error_lines)

    except (subprocess.TimeoutExpired, Exception):
        # If mypy fails, assume no errors detected
        n_errors = 0

    return MypyResult(n_errors=n_errors)


def evaluate_candidate_stdio(
    code: str, sample: AppsSample
) -> tuple[ExecutionResult, RuffResult, MypyResult]:
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        candidate_path = tmp_dir / "candidate.py"
        candidate_path.write_text(code)

        exec_result = run_tests_stdio(candidate_path, sample.tests)
        ruff_result = run_ruff(candidate_path)
        mypy_result = run_mypy(candidate_path)

    return exec_result, ruff_result, mypy_result


def evaluate_candidate_function(
    code: str, sample: AppsSample
) -> tuple[ExecutionResult, RuffResult, MypyResult]:
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        candidate_path = tmp_dir / "candidate.py"
        candidate_path.write_text(code)

        exec_result = run_tests_stdio(candidate_path, sample.tests)
        ruff_result = run_ruff(candidate_path)
        mypy_result = run_mypy(candidate_path)

    return exec_result, ruff_result, mypy_result
