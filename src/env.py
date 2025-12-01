import json
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


@contextmanager
def _temp_code_file(code: str):
    """Context manager that creates a temporary Python file with the given code."""
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        candidate_path = tmp_dir / "candidate.py"
        candidate_path.write_text(code)
        yield candidate_path


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
    syntax_error: bool
    stderr: str


@dataclass
class RuffResult:
    n_issues: int
    messages: list[str]


@dataclass
class MypyResult:
    n_errors: int
    messages: list[str]


def _run_single_test(
    candidate_path: Path, test_input: str, expected_output: str, timeout_s: float
) -> tuple[Verdict, str]:
    """Run a single test and return (verdict, stderr)."""
    expected_output = expected_output.strip()
    try:
        result = subprocess.run(
            ["python", str(candidate_path)],
            cwd=candidate_path.parent,
            capture_output=True,
            input=test_input,
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


def evaluate_unit_tests(
    code: str,
    tests: list[tuple[str, str]],
    max_workers: int | None = None,
    timeout_s: float = 5.0,
) -> ExecutionResult:
    with _temp_code_file(code) as candidate_path:
        n_total = len(tests)

        # Check for syntax errors first
        try:
            code = candidate_path.read_text()
            compile(code, str(candidate_path), "exec")
        except SyntaxError as e:
            return ExecutionResult(
                n_passed=0,
                n_total=n_total,
                syntax_error=True,
                stderr=str(e),
            )

        n_passed = 0
        stderr_output = ""
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        _run_single_test, candidate_path, inp, exp, timeout_s
                    )
                    for inp, exp in tests
                ]

                for future in as_completed(futures):
                    verdict, stderr = future.result()
                    if verdict is Verdict.AC:
                        n_passed += 1
                    if stderr:
                        stderr_output += stderr
        except Exception as e:
            print(f"Error during test execution: {e}")

        return ExecutionResult(
            n_passed=n_passed,
            n_total=n_total,
            syntax_error=False,
            stderr=stderr_output,
        )


def evaluate_ruff(code: str, select: list[str], ignore: list[str]) -> RuffResult:
    with _temp_code_file(code) as candidate_path:
        n_issues = 0
        messages: list[str] = []
        try:
            result = subprocess.run(
                [
                    "ruff",
                    "check",
                    "--select=" + ",".join(select),
                    "--ignore=" + ",".join(ignore),
                    "--output-format=json",
                    candidate_path.as_posix(),
                ],
                capture_output=True,
                text=True,
                timeout=10.0,
            )

            if result.stdout:
                issues = json.loads(result.stdout)
                n_issues = len(issues)
                messages = [issue["message"] for issue in issues]

        except Exception as e:
            print(f"Ruff error: {e}")

        return RuffResult(n_issues=n_issues, messages=messages)


def evaluate_mypy(code: str) -> MypyResult:
    with _temp_code_file(code) as candidate_path:
        n_errors = 0
        messages: list[str] = []
        try:
            result = subprocess.run(
                [
                    "mypy",
                    "--strict",
                    "--no-color-output",
                    "--no-error-summary",
                    candidate_path.as_posix(),
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
            messages = error_lines

        except Exception as e:
            print(f"Mypy error: {e}")

        return MypyResult(n_errors=n_errors, messages=messages)
