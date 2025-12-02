from .util import InferenceUtil
from datasets import Dataset, load_dataset

def get_system_prompt() -> str:
    return """You are Qwen, created by Alibaba Cloud. You are a helpful assistant.
You will be given a class and the skeleton of methods. Your task is to finish the implementation of the class and its methods.
Your output must be only Python code, no explanations, no comments, no markdown.
"""

def get_user_prompt(class_name: str, skeleton: str) -> str:
    instruction = f"Please complete the class {class_name} in the following code."
    instruction = instruction + '\n' + skeleton
    prompt = InferenceUtil.generate_prompt(instruction)
    return prompt


def question_to_prompt(class_name: str, skeleton: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": get_system_prompt()},
        {"role": "user", "content": get_user_prompt(class_name, skeleton)},
    ]


def load_classeval_holistic_dataset_prompt_only(
    split: str, max_samples: int | None = None
) -> Dataset:
    def map_function(row: dict) -> dict:
        return {
            "prompt": question_to_prompt(row["class_name"], row["skeleton"]),
            "tests": row["test"],
        }

    dataset: Dataset = load_dataset(
        "FudanSELab/ClassEval", split=split, trust_remote_code=True
    )
    dataset = dataset.select_columns(["class_name", "skeleton", "test"])
    dataset = dataset.map(
        map_function,
        remove_columns=["class_name", "skeleton", "test"],
        load_from_cache_file=False,
    )
    dataset = dataset.filter(lambda row: row["prompt"] != [] and row["tests"] != [])
    if max_samples is not None:
        dataset = dataset.select(range(min(len(dataset), max_samples)))
    print(f"Loaded {len(dataset)} samples from ClassEval dataset with prompt only.")
    return dataset