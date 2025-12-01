from dataclasses import dataclass

from .env import ExecutionResult, MypyResult, RuffResult


@dataclass
class RewardConfig:
    tests_weight: float = 1.0
    ruff_weight: float = 0.2
    mypy_weight: float = 0.2
    timeout_penalty: float = -1.0
    runtime_error_penalty: float = -0.5
    syntax_error_penalty: float = -1.0
    markdown_code_block_penalty: float = -0.1


def compute_reward(
    markdown_code_block: bool,
    exec_res: ExecutionResult,
    ruff_res: RuffResult,
    mypy_res: MypyResult,
    cfg: RewardConfig,
) -> float:
    if exec_res.syntax_error:
        return cfg.syntax_error_penalty

    if exec_res.runtime_error:
        return cfg.runtime_error_penalty

    tests_score = exec_res.n_passed / exec_res.n_total if exec_res.n_total > 0 else 0.0

    ruff_score = 1.0 / (1.0 + ruff_res.n_issues)

    mypy_score = 1.0 / (1.0 + mypy_res.n_errors)

    reward = (
        cfg.tests_weight * tests_score
        + cfg.ruff_weight * ruff_score
        + cfg.mypy_weight * mypy_score
    )

    if exec_res.timed_out:
        reward += cfg.timeout_penalty

    if markdown_code_block:
        reward += cfg.markdown_code_block_penalty

    return reward
