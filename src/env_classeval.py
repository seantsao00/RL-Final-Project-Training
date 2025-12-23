import json
import re
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from .eval_data import ClassEvalSample
from .env import MypyResult, RuffResult


@contextmanager
def _temp_code_file(code: str):
    """Context manager that creates a temporary Python file with the given code."""
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        candidate_path = tmp_dir / "candidate.py"
        candidate_path.write_text(code)
        yield candidate_path


@dataclass
class ClassEvalExecutionResult:
    n_passed: int
    n_total: int
    timed_out: bool
    runtime_error: bool
    syntax_error: bool
    stderr: str


def _check_syntax_error(code: str) -> SyntaxError | None:
    try:
        compile(code, "<string>", "exec")
        return None
    except SyntaxError as e:
        return e


def add_timeout_to_unittest_code(test_code: str, timeout_s: float = 2.0) -> str:
    """
    Wrap the unittest code to add a timeout with timeout_decorator to each test case.

    Args:
        test_code: The original unittest code as a string
        timeout_s: Timeout in seconds for each test

    Returns:
        Modified unittest code with timeouts
    """
    lines = test_code.split("\n")
    modified_lines = ["import timeout_decorator"]

    for line in lines:
        stripped_line = line.strip()
        if stripped_line.startswith("def test_"):
            indent = line[: line.index("def")]
            modified_lines.append(f"{indent}@timeout_decorator.timeout({timeout_s})")
        modified_lines.append(line)
    return "\n".join(modified_lines)


def build_full_class_code(
    class_name: str,
    import_statement: list[str],
    class_description: str,
    class_constructor: str,
    methods_info: list[dict],
    replaced_method: dict[str, str],
) -> str:
    """
    Build the full class code with all methods implemented.

    Args:
        class_name: Name of the class
        import_statement: List of import statements
        class_description: Class docstring/description
        class_constructor: Constructor code including class definition
        target_method_name: Name of the method being replaced
        methods_info: List of method info dicts with 'method_name' and 'solution_code'
        replaced_method: Dict with 'method_name': 'replace_code' for the replacement

    Returns:
        Complete class code as a string
    """
    parts = []

    if import_statement:
        parts.append("\n".join(import_statement))
        parts.append("")

    parts.append(f"class {class_name}:")
    if class_description:
        parts.append(class_description)

    constructor_lines = class_constructor.split("\n")
    for line in constructor_lines:
        if not line.strip().startswith("class "):
            parts.append(line)

    for method_info in methods_info:
        method_name = method_info["method_name"]

        if method_name in replaced_method:
            solution_code = replaced_method[method_name]
        else:
            solution_code = method_info["solution_code"]

        solution_lines = solution_code.split("\n")
        for line in solution_lines:
            if line.strip():
                parts.append(f"    {line}" if not line.startswith("    ") else line)
            else:
                parts.append("")
        parts.append("")

    return "\n".join(parts)


def run_unittest(full_test_code: str) -> ClassEvalExecutionResult:
    syntax_err = _check_syntax_error(full_test_code)
    if syntax_err:
        return ClassEvalExecutionResult(
            n_passed=0,
            n_total=0,
            timed_out=False,
            runtime_error=False,
            syntax_error=True,
            stderr=str(syntax_err),
        )

    with _temp_code_file(full_test_code) as candidate_path:
        try:
            res = subprocess.run(
                [
                    "python",
                    candidate_path.as_posix(),
                ],
                cwd=candidate_path.parent.as_posix(),
                capture_output=True,
                text=True,
                timeout=10.0,  # 10 second timeout per test file
            )
        except subprocess.TimeoutExpired:
            return ClassEvalExecutionResult(
                n_passed=0,
                n_total=0,
                timed_out=True,
                runtime_error=False,
                syntax_error=False,
                stderr="Test execution timed out after 10 seconds",
            )

        stderr_lines = res.stderr.splitlines(keepends=False)
        if not stderr_lines:
            # No output, likely a runtime error or empty test
            return ClassEvalExecutionResult(
                n_passed=0,
                n_total=0,
                timed_out=False,
                runtime_error=True,
                syntax_error=False,
                stderr=res.stderr,
            )
        
        result = stderr_lines[0]
        print(f"Unittest output: {result}")
        pruned_result = re.match(r"[.FE]*", result).group()
        if pruned_result != result:
            print("pruned_result:", pruned_result)
            result = pruned_result
        n_total = len(result)
        n_pass = result.count(".")

        return ClassEvalExecutionResult(
            n_passed=n_pass,
            n_total=n_total,
            timed_out=False,
            runtime_error=False,
            syntax_error=False,
            stderr=res.stderr,
        )


def evaluate_classeval_candidate(
    code: str,
    sample: ClassEvalSample,
    timeout_s: float = 10.0,
) -> ClassEvalExecutionResult:
    assembled_code = build_full_class_code(
        class_name=sample.class_name,
        import_statement=sample.import_statement,
        class_description="",
        class_constructor=sample.class_constructor,
        methods_info=sample.methods_info,
        replaced_method={sample.method_name: code},
    )

    unittest_code = f"""
import unittest

{sample.method_test_code}

if __name__ == "__main__":
    unittest.main()
"""
    unittest_code = add_timeout_to_unittest_code(unittest_code)

    full_test_code = f"""
{assembled_code}

{unittest_code}
"""
    return run_unittest(full_test_code)


def evaluate_ruff(
    assembled_code: str, select: list[str], ignore: list[str]
) -> RuffResult:
    syntax_err = _check_syntax_error(assembled_code)
    if syntax_err:
        return RuffResult(0, [str(syntax_err)], True)

    with _temp_code_file(assembled_code) as candidate_path:
        n_issues = 0
        messages: list[str] = []
        try:
            result = subprocess.run(
                [
                    "ruff",
                    "check",
                    "--select=" + ",".join(select),
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


def evaluate_mypy(
    assembled_code: str,
) -> MypyResult:
    syntax_err = _check_syntax_error(assembled_code)
    if syntax_err:
        return MypyResult(0, [str(syntax_err)], syntax_error=True)

    with _temp_code_file(assembled_code) as candidate_path:
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

        return MypyResult(n_errors=n_errors, messages=messages, syntax_error=False)