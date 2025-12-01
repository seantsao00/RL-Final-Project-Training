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


def _check_syntax_error(code: str) -> SyntaxError | None:
    try:
        compile(code, "<string>", "exec")
        return None
    except SyntaxError as e:
        return e


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
    stderr: str
    syntax_error: bool = False


@dataclass
class RuffResult:
    n_issues: int
    messages: list[str]
    syntax_error: bool = False


@dataclass
class MypyResult:
    n_errors: int
    messages: list[str]
    syntax_error: bool = False


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

        syntax_err = _check_syntax_error(candidate_path.read_text())
        if syntax_err:
            return ExecutionResult(0, n_total, str(syntax_err), syntax_error=True)

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

        return ExecutionResult(n_passed, n_total, stderr_output)


def evaluate_ruff(code: str, select: list[str], ignore: list[str]) -> RuffResult:
    with _temp_code_file(code) as candidate_path:
        # Check for syntax errors first
        syntax_err = _check_syntax_error(candidate_path.read_text())
        if syntax_err:
            return RuffResult(0, [str(syntax_err)], True)

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

        return RuffResult(n_issues, messages)


def evaluate_mypy(code: str) -> MypyResult:
    with _temp_code_file(code) as candidate_path:
        # Check for syntax errors first
        syntax_err = _check_syntax_error(candidate_path.read_text())
        if syntax_err:
            return MypyResult(0, [str(syntax_err)], syntax_error=True)

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

        return MypyResult(n_errors, messages)
