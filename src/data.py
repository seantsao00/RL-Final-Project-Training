import json
import sys
from dataclasses import dataclass
from pathlib import Path

from datasets import Dataset, load_dataset

# To prevent issues with large integers in test cases
sys.set_int_max_str_digits(0)


@dataclass
class ClassEvalSample:
    task_id: str
    method_name: str
    import_statement: list[str]
    class_name: str
    # skeleton: str
    # class_description: str
    class_test_code: str
    method_test_code: str
    class_constructor: str
    methods_info: list[dict[str, any]]
    # solution_code: str


def load_apps_dataset_prompt_only(
    split: str, max_samples: int | None = None
) -> Dataset:
    def map_function(row: dict) -> dict:
        return {
            "prompt": question_to_prompt(row["question"]),
            "tests": build_tests(row["input_output"]),
        }

    dataset: Dataset = load_dataset(
        "codeparrot/apps", split=split, trust_remote_code=True
    )  # type: ignore
    dataset = dataset.filter(lambda row: row["difficulty"] in ["introductory"])
    dataset = dataset.select_columns(["question", "input_output"])
    dataset = dataset.map(
        map_function,
        remove_columns=["question", "input_output"],
        load_from_cache_file=False,
    )
    dataset = dataset.filter(lambda row: row["prompt"] != [] and row["tests"] != [])
    if max_samples is not None:
        dataset = dataset.select(range(min(len(dataset), max_samples)))
    print(f"Loaded {len(dataset)} samples from Apps dataset with prompt only.")
    return dataset


def load_classeval_dataset(
    max_samples: int | None = None,
    path: Path = (
        Path(__file__).resolve().parents[1]
        / "ClassEval"
        / "data"
        / "ClassEval_data.json"
    ),
) -> Dataset:
    """
    This function loads the ClassEval dataset and returns a `datasets.Dataset` object.
    It keeps all original fields to support Incremental and Compositional generation strategies.
    """

    dataset: Dataset
    if path is None:
        dataset = load_dataset(
            "FudanSELab/ClassEval", split="test", trust_remote_code=True
        )
    else:
        dataset = load_dataset(
            "json",
            data_files=str(path),
        )

    def map_prompt(row: dict) -> dict:
        mapped = {
            "prompt": get_prompt_only_conversational_prompt(
                row["skeleton"], dataset_name="classeval"
            ),
        }
        return mapped

    dataset = dataset.map(
        map_prompt,
        load_from_cache_file=False,
    )

    dataset = dataset.filter(lambda row: row["prompt"] != [])

    if max_samples is not None:
        dataset = dataset.select(range(min(len(dataset), max_samples)))

    return dataset


def load_classeval_dataset_prompt_only(
    max_samples: int | None = None,
    path: Path = (
        Path(__file__).resolve().parents[1]
        / "ClassEval"
        / "data"
        / "ClassEval_data.json"
    ),
) -> Dataset:
    """
    Load the ClassEval dataset in compositional format.
    Each row represents one method to complete, given the class skeleton with other methods implemented.
    Returns a dataset with columns: prompt, task_id, method_name, ground_truth, test_code
    """
    dataset: Dataset
    if path is None:
        dataset = load_dataset(
            "FudanSELab/ClassEval", split="test", trust_remote_code=True
        )
    else:
        dataset = load_dataset(
            "json",
            data_files=str(path),
        )["train"]

    compositional_rows = []

    for class_data in dataset:
        task_id = class_data["task_id"]
        class_name = class_data["class_name"]
        import_statement = class_data["import_statement"]
        class_description = class_data["class_description"]
        class_constructor = class_data["class_constructor"]
        class_test_code = class_data["test_code"]
        methods_info = class_data["methods_info"]
        skeleton = class_data["skeleton"]

        for target_method_info in methods_info:
            target_method_name = target_method_info["method_name"]
            target_test_code = target_method_info["test_code"]

            # Create conversational prompt
            prompt = get_classeval_compositional_prompt(
                skeleton, target_method_name, class_name
            )

            # Create row
            row = {
                "prompt": prompt,
                "task_id": f"{task_id}_{target_method_name}",
                "method_name": target_method_name,
                "import_statement": import_statement,
                "class_name": class_name,
                "class_test_code": class_test_code,
                "method_test_code": target_test_code,
                "class_constructor": class_constructor,
                "methods_info": methods_info,
            }
            compositional_rows.append(row)

    dataset = Dataset.from_list(compositional_rows)

    if max_samples is not None:
        dataset = dataset.select(range(min(len(dataset), max_samples)))

    return dataset


def get_system_prompt(dataset_name="apps") -> str:
    if dataset_name == "apps":
        return """You are Qwen, created by Alibaba Cloud. You are a helpful assistant.
You will be given a programming question and you must provide a solution in Python. 
Your output must be only Python code, no explanations, no comments, no markdown.
The program must be a standalone solution using only the Python standard library.
The program should read input exactly as described.
The program should print only the required output.
"""
    if dataset_name == "classeval":
        return """You are an exceptionally intelligent Python class coding assistant.
Given a class skeleton with docstrings and method signatures, write the complete implementation of the pointed method in pure Python.
Your output must be only valid Python code, no explanations, no comments, no docstring, no markdown.
The implementation should adhere to the provided method signatures and docstrings.
"""


# The class should be self-contained and use only the Python standard library.


def get_user_prompt(question: str) -> str:
    return f"""Question:
{question}
"""


def question_to_prompt(question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": get_system_prompt("apps")},
        {
            "role": "user",
            "content": get_user_prompt(
                question,
            ),
        },
    ]


def get_classeval_compositional_prompt(
    skeleton_code: str, target_method_name: str, class_name: str
) -> list[dict[str, str]]:
    """
    Create a conversational prompt for compositional method completion.
    """
    system_prompt = get_system_prompt("classeval")

    user_prompt = f"""Provides the complete code without docstring for `{target_method_name}` in the class `{class_name}`.

Class code:
```python
{skeleton_code}

```

Provide only the complete implementation of the `{target_method_name}` method (properly indented):"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_tests(input_output: str) -> list[tuple[str, str]]:
    """
    Build test cases from the input_output JSON string.

    Returns an empty list if any error occurs.
    """
    try:
        tests: list[tuple[str, str]] = []
        test_data: dict[str, list[str]] = json.loads(input_output)
        assert isinstance(test_data, dict)
        inputs: list[str] = test_data["inputs"]
        assert isinstance(inputs, list)
        outputs: list[str] = test_data["outputs"]
        assert isinstance(outputs, list)
        for inp, out in zip(inputs, outputs, strict=True):
            assert isinstance(inp, str)
            assert isinstance(out, str)
            tests.append((inp, out))
        return tests
    except (AssertionError, KeyError, json.JSONDecodeError):
        return []
