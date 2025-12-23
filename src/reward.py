import re
from dataclasses import dataclass

from .env import evaluate_mypy, evaluate_ruff, evaluate_unit_tests


@dataclass
class RewardConfig:
    tests_weight: float
    ruff_weight: float
    mypy_weight: float
    syntax_error_penalty: float


def _extract_code(completion: str) -> str:
    """Extract code from markdown code block if present."""
    match = re.search(r"```python(.*?)```", completion, re.DOTALL)
    return match.group(1).strip() if match else completion



def ruff_reward_function(
    prompts: list[list[dict[str, str]]],
    completions: list[list[dict[str, str]]],
    **kwargs,
) -> list[float]:
    solutions = [_extract_code(comp[0]["content"]) for comp in completions]

    rewards: list[float] = []
    for i, solution in enumerate(solutions):
        result = evaluate_ruff(solution)
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

def ablation_ruff_reward_function(
    prompts: list[list[dict[str, str]]],
    completions: list[list[dict[str, str]]],
    **kwargs,
) -> list[float]:
    solutions = [_extract_code(comp[0]["content"]) for comp in completions]

    rewards: list[float] = []
    for i, solution in enumerate(solutions):
        result = evaluate_ruff(solution)
        # reward = 1 / (1.0 + result.n_issues)
        reward = 1 if result.n_issues == 0 else 0
        rewards.append(reward)

        if i == 0:
            print("Ruff Reward Debug Info:")
            print(f"Ruff result: {result}")
            print(f"Calculated reward: {reward}")
            print("================================")

    return rewards


def ablation_mypy_reward_function(
    prompts: list[list[dict[str, str]]],
    completions: list[list[dict[str, str]]],
    **kwargs,
) -> list[float]:
    solutions = [_extract_code(comp[0]["content"]) for comp in completions]

    rewards: list[float] = []
    for i, solution in enumerate(solutions):
        result = evaluate_mypy(solution)
        # reward = 1 / (1.0 + result.n_errors)
        reward = 1 if result.n_errors == 0 else 0
        rewards.append(reward)

        if i == 0:
            print("Mypy Reward Debug Info:")
            print(f"Mypy result: {result}")
            print(f"Calculated reward: {reward}")
            print("================================")

    return rewards
