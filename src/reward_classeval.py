import re
from dataclasses import dataclass

from .env_classeval import (
    evaluate_mypy,
    evaluate_ruff,
    evaluate_classeval_candidate,
)
from .data import ClassEvalSample


@dataclass
class RewardConfig:
    tests_weight: float = 1.0
    ruff_weight: float = 0.2
    mypy_weight: float = 0.2
    syntax_error_penalty: float = -1.0


def _extract_code(completion: str) -> str:
    """Extract code from markdown code block if present."""
    match = re.search(r"```python(.*?)```", completion, re.DOTALL)
    return match.group(1).strip() if match else completion

def classeval_unittest_reward_function(
    prompts: list[list[dict[str, str]]],
    completions: list[list[dict[str, str]]],
    syntax_error_penalty: float,
    **kwargs,
) -> list[float]:
    """Compute unit-test rewards for Classeval compositional samples.

    Expects the following dataset fields in kwargs (each is a list aligned with batch rows):
    - import_statement, class_name, class_constructor, methods_info, method_name
    """
    solutions = [_extract_code(comp[0]["content"]) for comp in completions]

    import_statements: list[list[str]] = kwargs["import_statement"]
    class_names: list[str] = kwargs["class_name"]
    class_constructors: list[str] = kwargs["class_constructor"]
    methods_infos: list[list[dict]] = kwargs["methods_info"]
    method_names: list[str] = kwargs["method_name"]

    rewards: list[float] = []

    for i, solution in enumerate(solutions):
        sample = ClassEvalSample(
            task_id="",
            method_name=method_names[i],
            import_statement=import_statements[i],
            class_name=class_names[i],
            class_test_code="",
            method_test_code=kwargs["method_test_code"][i],
            class_constructor=class_constructors[i],
            methods_info=methods_infos[i],
        )
        result = evaluate_classeval_candidate(solution, sample)
        if result.syntax_error:
            reward = syntax_error_penalty
        else:
            reward = result.n_passed / result.n_total if result.n_total > 0 else 0.0
        rewards.append(reward)

        if i == 0:
            print("Classeval Unit Test Reward Debug Info:")
            for prompt in prompts[0]:
                print(f"{prompt['role']}:\n{prompt['content']}\n")
            print("================================")
            for completion in completions[0]:
                print(f"{completion['role']}:\n{completion['content']}\n")
            print("================================")
            print(f"Tests result: {result}")
            print(f"Calculated reward: {reward}")

    return rewards


def ruff_reward_function(
    prompts: list[list[dict[str, str]]],
    completions: list[list[dict[str, str]]],
    **kwargs,
) -> list[float]:
    solutions = [_extract_code(comp[0]["content"]) for comp in completions]

    rewards: list[float] = []
    for i, solution in enumerate(solutions):
        # Build sample for proper code assembly within ruff
        sample = ClassEvalSample(
            task_id="",
            method_name=kwargs["method_name"][i],
            import_statement=kwargs["import_statement"][i],
            class_name=kwargs["class_name"][i],
            class_test_code="",
            method_test_code=kwargs["method_test_code"][i],
            class_constructor=kwargs["class_constructor"][i],
            methods_info=kwargs["methods_info"][i],
        )
        result = evaluate_ruff(solution, sample)
        reward = 1 / (1.0 + result.n_issues)
        rewards.append(reward)

        if i == 0:
            print("Ruff Reward Debug Info:")
            print(f"Ruff result: {result}")
            print(f"Calculated reward: {reward}")

    return rewards


def mypy_reward_function(
    prompts: list[list[dict[str, str]]],
    completions: list[list[dict[str, str]]],
    **kwargs,
) -> list[float]:
    solutions = [_extract_code(comp[0]["content"]) for comp in completions]

    rewards: list[float] = []
    for i, solution in enumerate(solutions):
        sample = ClassEvalSample(
            task_id="",
            method_name=kwargs["method_name"][i],
            import_statement=kwargs["import_statement"][i],
            class_name=kwargs["class_name"][i],
            class_test_code="",
            method_test_code=kwargs["method_test_code"][i],
            class_constructor=kwargs["class_constructor"][i],
            methods_info=kwargs["methods_info"][i],
        )
        result = evaluate_mypy(solution, sample)
        reward = 1 / (1.0 + result.n_errors)
        rewards.append(reward)

        if i == 0:
            print("Mypy Reward Debug Info:")
            print(f"Mypy result: {result}")
            print(f"Calculated reward: {reward}")

    return rewards
