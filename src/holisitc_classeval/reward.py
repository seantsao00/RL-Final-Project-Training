import re
from .env import evaluate_unit_tests


def _extract_code(completion: str) -> str:
    """Extract code from markdown code block if present."""
    match = re.search(r"```python(.*?)```", completion, re.DOTALL)
    # print(match)
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

    for i, (solution, test_script) in enumerate(zip(solutions, tests, strict=True)):
        test_code_py = solution + '\n' + test_script
        result = evaluate_unit_tests(test_code_py)
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