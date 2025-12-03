import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from .env import RuffResult, MypyResult
from .data import ClassEvalSample


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

    full_test_code = f"""
{assembled_code}

{unittest_code}
"""

    try:
        compile(full_test_code, "<string>", "exec")
    except Exception as e:
        return ClassEvalExecutionResult(
            n_passed=0,
            n_total=0,
            timed_out=False,
            runtime_error=False,
            syntax_error=True,
            stderr=str(e),
        )

    with _temp_code_file(code) as candidate_path:
        test_file = candidate_path.read_text()

        try:
            result = subprocess.run(
                ["python", "-m", "unittest", test_file],
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )

            stderr = result.stderr
            stdout = result.stdout

            # Parse unittest output to count tests
            # unittest outputs like "Ran X tests" or similar
            n_total = 0
            n_passed = 0

            # Try to parse from stderr (unittest outputs to stderr by default)
            output = stderr + stdout

            if "Ran" in output:
                for line in output.split("\n"):
                    if line.startswith("Ran "):
                        # Extract number from "Ran X test(s)"
                        parts = line.split()
                        if len(parts) >= 2 and parts[1].isdigit():
                            n_total = int(parts[1])
                        break

            # Check if all tests passed
            if result.returncode == 0:
                n_passed = n_total
            else:
                # Parse failures/errors
                if "FAILED" in output:
                    # Look for patterns like "failures=X, errors=Y"
                    failures = 0
                    errors = 0
                    for line in output.split("\n"):
                        if "failures=" in line.lower():
                            try:
                                failures = int(
                                    line.split("failures=")[1]
                                    .split(",")[0]
                                    .split(")")[0]
                                    .strip()
                                )
                            except:
                                pass
                        if "errors=" in line.lower():
                            try:
                                errors = int(
                                    line.split("errors=")[1]
                                    .split(",")[0]
                                    .split(")")[0]
                                    .strip()
                                )
                            except:
                                pass

                    n_passed = max(0, n_total - failures - errors)

            return ClassEvalExecutionResult(
                n_passed=n_passed,
                n_total=n_total,
                timed_out=False,
                runtime_error=result.returncode != 0 and "Error" in output,
                syntax_error=False,
                stderr=stderr,
            )

        except subprocess.TimeoutExpired:
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


def evaluate_ruff(
    code: str,
    sample: ClassEvalSample,
) -> RuffResult:
    assembled_code = build_full_class_code(
        class_name=sample.class_name,
        import_statement=sample.import_statement,
        class_description="",
        class_constructor=sample.class_constructor,
        methods_info=sample.methods_info,
        replaced_method={sample.method_name: code},
    )
    with _temp_code_file(assembled_code) as candidate_path:
        n_issues = 0
        messages: list[str] = []
        try:
            result = subprocess.run(
                [
                    "ruff",
                    "check",
                    "--select=F,W,E,UP,C4,FA,ISC,RET,SIM,TID,TC,PTH,TD,NPY",
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
    code: str,
    sample: ClassEvalSample,
) -> MypyResult:
    assembled_code = build_full_class_code(
        class_name=sample.class_name,
        import_statement=sample.import_statement,
        class_description="",
        class_constructor=sample.class_constructor,
        methods_info=sample.methods_info,
        replaced_method={sample.method_name: code},
    )
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

        return MypyResult(n_errors=n_errors, messages=messages)
