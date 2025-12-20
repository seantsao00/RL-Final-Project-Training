import json
import sys

from datasets import Dataset, load_dataset

# To prevent issues with large integers in test cases
sys.set_int_max_str_digits(0)


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


def get_system_prompt() -> str:
    return """You are Qwen, created by Alibaba Cloud. You are a helpful assistant.
You will be given a programming question and you must provide a solution in Python. 
Your output must be only Python code, no explanations, no comments, no markdown.
The program must be a standalone solution using only the Python standard library.
The program should read input exactly as described.
The program should print only the required output.
"""


def get_user_prompt(question: str) -> str:
    return f"""Question:
{question}
"""


def question_to_prompt(question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": get_system_prompt()},
        {"role": "user", "content": get_user_prompt(question)},
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
