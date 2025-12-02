import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import unittest
import io
import uuid
import sys
import shutil
import os

@contextmanager
def _temp_code_file(code: str):
    """Context manager that creates a temporary Python file with the given code."""
    original_cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        candidate_path = tmp_dir / f"candidate_{uuid.uuid4().hex}.py"
        candidate_path.write_text(code)
        module_name = candidate_path.stem
        try:
            # Change to temp directory so any files created by tests go there
            os.chdir(tmp_dir)
            yield candidate_path
        finally:
            # Restore original working directory
            os.chdir(original_cwd)
            # Clean up any cached modules to prevent memory leaks
            modules_to_remove = [m for m in sys.modules if m == module_name or m.startswith(f"{module_name}.")]
            for mod in modules_to_remove:
                del sys.modules[mod]
            # Clean up __pycache__ if created
            pycache_dir = tmp_dir / "__pycache__"
            if pycache_dir.exists():
                shutil.rmtree(pycache_dir)

@dataclass
class ExecutionResult:
    n_passed: int
    n_total: int
    syntax_error: bool
    stderr: str


def evaluate_unit_tests(code: str) -> ExecutionResult:
    with _temp_code_file(code) as candidate_path:
        loader = unittest.TestLoader()
        suite = loader.discover(
            start_dir=str(candidate_path.parent),
            pattern=candidate_path.name
        )
        n_total = suite.countTestCases()
        # print(code, n_total)

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
        
        stderr = io.StringIO()
        runner = unittest.TextTestRunner(verbosity=2, stream=stderr)
        test_result = runner.run(suite)

        failures = len(test_result.failures)
        errors = len(test_result.errors)

        n_passed = n_total - failures - errors
        stderr_output = stderr.getvalue()

        return ExecutionResult(
            n_passed=n_passed,
            n_total=n_total,
            syntax_error=False,
            stderr=stderr_output,
        )