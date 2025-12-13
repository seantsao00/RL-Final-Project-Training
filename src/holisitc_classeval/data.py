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
    split: str, max_samples: int | None = None, train_test_split_ratio: float = 0.8
) -> Dataset:
    """
    Load ClassEval dataset and split into train/test.
    
    Args:
        split: 'train' or 'test' to return that portion
        max_samples: Maximum number of samples to use from the full dataset before splitting
        train_test_split_ratio: Ratio of train samples (default 0.8 means 80% train, 20% test)
    """
    def map_function(row: dict) -> dict:
        return {
            "prompt": question_to_prompt(row["class_name"], row["skeleton"]),
            "tests": row["test"],
        }

    # Load the full test split from HuggingFace (the only split available)
    dataset: Dataset = load_dataset(
        "FudanSELab/ClassEval", split="test", trust_remote_code=True
    )
    dataset = dataset.select_columns(["class_name", "skeleton", "test"])
    dataset = dataset.map(
        map_function,
        remove_columns=["class_name", "skeleton", "test"],
        load_from_cache_file=False,
    )
    dataset = dataset.filter(lambda row: row["prompt"] != [] and row["tests"] != [])
    
    # Apply max_samples to the full dataset before splitting
    if max_samples is not None:
        dataset = dataset.select(range(min(len(dataset), max_samples)))
    
    # Split the dataset into train and test
    total_samples = len(dataset)
    train_size = int(total_samples * train_test_split_ratio)
    
    if split == "train":
        dataset = dataset.select(range(train_size))
        print(f"Loaded {len(dataset)} train samples from ClassEval dataset.")
    elif split == "test":
        dataset = dataset.select(range(train_size, total_samples))
        print(f"Loaded {len(dataset)} test samples from ClassEval dataset.")
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train' or 'test'.")
    
    return dataset