import re
from dataclasses import dataclass

from .env import evaluate_mypy, evaluate_ruff, evaluate_unit_tests


@dataclass
class RewardConfig:
    tests_weight: float
    ruff_weight: float
    mypy_weight: float
    syntax_error_penalty: float
    ruff_select: list[str]
    ruff_ignore: list[str]


def _extract_code(completion: str) -> str:
    """Extract code from markdown code block if present."""
    match = re.search(r"```python(.*?)```", completion, re.DOTALL)
    return match.group(1).strip() if match else completion


def unit_test_reward_function(
    prompts: list[list[dict[str, str]]],
    completions: list[list[dict[str, str]]],
    syntax_error_penalty: float,
    test_threads: int | None = None,
    **kwargs,
) -> list[float]:
    solutions = [_extract_code(comp[0]["content"]) for comp in completions]
    # Assume that a field "tests" exists in dataset samples
    tests: list[list[tuple[str, str]]] = kwargs["tests"]

    rewards: list[float] = []
    for i, (solution, test_cases) in enumerate(zip(solutions, tests, strict=True)):
        result = evaluate_unit_tests(solution, test_cases, test_threads)
        if result.syntax_error:
            reward = syntax_error_penalty
        else:
            reward = result.n_passed / result.n_total if result.n_total > 0 else 0.0
        rewards.append(reward)

        if i == 0:
            print("Unit Test Reward Debug Info:")
            for prompt in prompts[0]:
                print(f"{prompt['role']}:\n{prompt['content']}\n")
            print("================================")
            for completion in completions[0]:
                print(f"{completion['role']}:\n{completion['content']}\n")
            print("================================")
            print(f"Tests result: {result}")
            print(f"Calculated reward: {reward}")
            print("================================")

    return rewards


def ruff_reward_function(
    prompts: list[list[dict[str, str]]],
    completions: list[list[dict[str, str]]],
    ruff_select: list[str],
    ruff_ignore: list[str],
    **kwargs,
) -> list[float]:
    solutions = [_extract_code(comp[0]["content"]) for comp in completions]

    rewards: list[float] = []
    for i, solution in enumerate(solutions):
        result = evaluate_ruff(solution, select=ruff_select, ignore=ruff_ignore)
        reward = 1 / (1.0 + result.n_issues)
        rewards.append(reward)

        if i == 0:
            print("Ruff Reward Debug Info:")
            print(f"Ruff result: {result}")
            print(f"Calculated reward: {reward}")
            print("================================")

    return rewards


def mypy_reward_function(
    prompts: list[list[dict[str, str]]],
    completions: list[list[dict[str, str]]],
    **kwargs,
) -> list[float]:
    solutions = [_extract_code(comp[0]["content"]) for comp in completions]

    rewards: list[float] = []
    for i, solution in enumerate(solutions):
        result = evaluate_mypy(solution)
        reward = 1 / (1.0 + result.n_errors)
        rewards.append(reward)

        if i == 0:
            print("Mypy Reward Debug Info:")
            print(f"Mypy result: {result}")
            print(f"Calculated reward: {reward}")
            print("================================")

    return rewards
