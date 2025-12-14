import re
from dataclasses import dataclass
from functools import partial, update_wrapper

from .env_classeval import (
    evaluate_mypy,
    evaluate_ruff,
    evaluate_classeval_candidate,
)
from .data import ClassEvalSample
from .env_classeval import (
    build_full_class_code,
    ClassEvalExecutionResult,
    MypyResult,
    RuffResult,
)


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
    pattern_list = [
        r"```python(.*?)```",
        r"```ruby(.*?)```",
        r"```scss(.*?)```",
        r"```python(.*?)",
        r"```(.*?)```",
        r"\[PYTHON\](.*?)\[/PYTHON\]",
    ]
    # match = re.search(r"```python(.*?)```", completion, re.DOTALL)
    # return match.group(1).strip() if match else completion
    for pattern in pattern_list:
        try:
            code = re.findall(pattern, completion, re.S)[0]
            return code
        except:
            continue
    return completion


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
        result: ClassEvalExecutionResult = evaluate_classeval_candidate(
            solution, sample
        )
        if result.syntax_error:
            reward = syntax_error_penalty
        else:
            reward = result.n_passed / result.n_total if result.n_total > 0 else 0.0
        rewards.append(reward)

        if i == 0:
            print("Classeval Unit Test Reward Debug Info:")
            # for prompt in prompts[0]:
            #     print(f"{prompt['role']}:\n{prompt['content']}\n")
            # print("================================")
            # for completion in completions[0]:
            #     print(f"{completion['role']}:\n{completion['content']}\n")
            # print("================================")
        print(f"Tests result: {result}")
        print(f"Calculated reward: {reward}")

    return rewards


def ruff_reward_function(
    prompts: list[list[dict[str, str]]],
    completions: list[list[dict[str, str]]],
    syntax_error_penalty: float,
    ruff_select: list[str],
    ruff_ignore: list[str],
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
        assembled_code = build_full_class_code(
            class_name=sample.class_name,
            import_statement=sample.import_statement,
            class_description="",
            class_constructor=sample.class_constructor,
            methods_info=sample.methods_info,
            replaced_method={sample.method_name: solution},
        )
        result: RuffResult = evaluate_ruff(assembled_code, ruff_select, ruff_ignore)
        if result.syntax_error:
            reward = syntax_error_penalty
        else:
            base_code = build_full_class_code(
                class_name=sample.class_name,
                import_statement=sample.import_statement,
                class_description="",
                class_constructor=sample.class_constructor,
                methods_info=sample.methods_info,
                replaced_method={
                    sample.method_name: f"    def {sample.method_name}(self):\n        pass\n"
                },
            )
            base_result = evaluate_ruff(base_code, ruff_select, ruff_ignore)
        reward = 1 / (1.0 + max(result.n_issues - base_result.n_issues, 0))
        rewards.append(reward)

        if i == 0:
            print("Ruff Reward Debug Info:")
            print(f"Ruff result: {result}")
            print(f"Calculated reward: {reward}")

    return rewards


def mypy_reward_function(
    prompts: list[list[dict[str, str]]],
    completions: list[list[dict[str, str]]],
    syntax_error_penalty: float,
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
        assembled_code = build_full_class_code(
            class_name=sample.class_name,
            import_statement=sample.import_statement,
            class_description="",
            class_constructor=sample.class_constructor,
            methods_info=sample.methods_info,
            replaced_method={sample.method_name: solution},
        )
        result: MypyResult = evaluate_mypy(assembled_code)
        if result.syntax_error:
            reward = syntax_error_penalty
        else:
            base_code = build_full_class_code(
                class_name=sample.class_name,
                import_statement=sample.import_statement,
                class_description="",
                class_constructor=sample.class_constructor,
                methods_info=sample.methods_info,
                replaced_method={
                    sample.method_name: f"    def {sample.method_name}(self):\n        pass\n"
                },
            )
            base_result = evaluate_mypy(base_code)
            reward = 1 / (1.0 + max(result.n_errors - base_result.n_errors, 0))
        rewards.append(reward)

        if i == 0:
            print("Mypy Reward Debug Info:")
            print(f"Mypy result: {result}")
            print(f"Calculated reward: {reward}")

    return rewards

def create_reward_funcs(
    reward_cfg: RewardConfig
) -> list:
    return [
        update_wrapper(
            partial(
                classeval_unittest_reward_function,
                syntax_error_penalty=reward_cfg.syntax_error_penalty,
            ),
            classeval_unittest_reward_function,
        ),
        update_wrapper(
            partial(
                ruff_reward_function,
                syntax_error_penalty=reward_cfg.syntax_error_penalty,
                ruff_select=reward_cfg.ruff_select,
                ruff_ignore=reward_cfg.ruff_ignore,
            ),
            ruff_reward_function,
        ),
        update_wrapper(
            partial(
                mypy_reward_function,
                syntax_error_penalty=reward_cfg.syntax_error_penalty,
            ),
            mypy_reward_function,
        ),
    ]
