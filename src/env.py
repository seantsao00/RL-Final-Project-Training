import json
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .data import AppsSample


class Verdict(Enum):
    AC = "Accepted"
    WA = "Wrong Answer"
    TLE = "Time Limit Exceeded"
    RE = "Runtime Error"
    CE = "Compilation Error"


@dataclass
class ExecutionResult:
    n_passed: int
    n_total: int
    timed_out: bool
    runtime_error: bool
    syntax_error: bool
    stderr: str


@dataclass
class RuffResult:
    n_issues: int


@dataclass
class MypyResult:
    n_errors: int


def _run_single_test(
    candidate_path: Path, test_input: str, expected_output: str, timeout_s: float
) -> tuple[Verdict, str]:
    """Run a single test and return (verdict, stderr)."""
    expected_output = expected_output.strip()
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
            return Verdict.RE, stderr
        elif actual_output == expected_output:
            return Verdict.AC, stderr
        else:
            return Verdict.WA, stderr

    except subprocess.TimeoutExpired:
        return Verdict.TLE, ""
    except Exception as e:
        return Verdict.RE, f"Exception: {str(e)}"


def run_tests(
    candidate_path: Path,
    tests: list[tuple[str, str]],
    timeout_s: float = 5.0,
    max_workers: int | None = None,
) -> ExecutionResult:
    n_total = len(tests)

    if n_total == 0:
        return ExecutionResult(
            n_passed=0,
            n_total=0,
            timed_out=False,
            runtime_error=False,
            syntax_error=False,
            stderr="",
        )

    # Check for syntax errors first
    try:
        code = candidate_path.read_text()
        compile(code, str(candidate_path), "exec")
    except SyntaxError as e:
        return ExecutionResult(
            n_passed=0,
            n_total=n_total,
            timed_out=False,
            runtime_error=False,
            syntax_error=True,
            stderr=str(e),
        )
    except Exception as e:
        return ExecutionResult(
            n_passed=0,
            n_total=n_total,
            timed_out=False,
            runtime_error=False,
            syntax_error=True,
            stderr=str(e),
        )

    n_passed = 0
    timed_out = False
    runtime_error = False
    stderr_output = ""

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_run_single_test, candidate_path, inp, exp, timeout_s): i
            for i, (inp, exp) in enumerate(tests)
        }

        for future in as_completed(futures):
            verdict, stderr = future.result()
            if verdict is Verdict.AC:
                n_passed += 1
            elif verdict is Verdict.TLE:
                timed_out = True
            elif verdict is Verdict.RE:
                runtime_error = True
            if stderr:
                stderr_output += stderr

    return ExecutionResult(
        n_passed=n_passed,
        n_total=n_total,
        timed_out=timed_out,
        runtime_error=runtime_error,
        syntax_error=False,
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


def evaluate_candidate(
    code: str,
    sample: AppsSample,
    max_workers: int | None = None,
) -> tuple[ExecutionResult, RuffResult, MypyResult]:
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        candidate_path = tmp_dir / "candidate.py"
        candidate_path.write_text(code)

        exec_result = run_tests(candidate_path, sample.tests, max_workers=max_workers)
        ruff_result = run_ruff(candidate_path)
        mypy_result = run_mypy(candidate_path)

    return exec_result, ruff_result, mypy_result
