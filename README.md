This branch is used to perform cross-dataset evaluation on the [HumanEval](https://github.com/openai/human-eval) dataset.

- `human-eval/` directory: contains original data cloned from official GitHub page.
- `analyze.py`: The code used to analyze and calculate the syntax error, Ruff error, and mypy error from an existed response file.
- `apps_eval.py`: The code used to generate and evaluate the responses of [APPS](https://github.com/hendrycks/apps/tree/main) dataset.
- `checkpoint_eval.py`: The code used to generate responses of HumanEval dataset from QRCoder trained model.
- `official_baseline_eval.py`: The code used to generate responses of HumanEval dataset from baseline qwen2.5-Coder model.

The other code could be the same as main branch.
