# QRCode ClassEval-Holistic

This part is for the training and evaluation part for ClassEval-Holistic In-Dataset evaluation and the ablation study on our reward design and 0/1 reward.

## Environment Setup

### install uv and the projects' packages

```
pip install uv
uv sync
```

## Training

For 3B model: (Need at least 32GB GPU memory)
```
uv run python -m src.classeval_holistic_train --config configs/holistic_classeval_3b.yaml
```

For 3B model with ablation study (0/1 reward):

```
uv run python -m src.classeval_holistic_train --config configs/holistic_classeval_3b_ablation.yaml
```


For 7B model: (Need at least 48GB GPU memory)

```
uv run python -m src.classeval_holistic_train --config configs/holistic_classeval_7b.yaml
```

For 7B model with ablation study (0/1 reward):

```
uv run python -m src.classeval_holistic_train --config configs/holistic_classeval_7b_ablation.yaml
``` 

## Evaluation

```
uv run python -m src.classeval_holistic_evaluate --model_path checkpoints/grpo-classeval-holistic/
uv run python -m process_json_for_eval --json_file evaluation_results/grpo-classeval-holistic_evaluation.json --output_dir results/grpo-classeval-holistic
uv run python -m evaluate --eval_output_dir results/grpo-classeval-holistic/
```

You will see things like:

```
Unittest output: FFFFF......FFFFFF..FFFFF.
Class: SQLQueryBuilder, Ruff issues: 4, Mypy errors: 0
.
<Ignore this part>
.
Class: ZipFileProcessor, Ruff issues: 1, Mypy errors: 0
testcase passrate: 0.6363636363636364 (294/462)
total syntax errors: 0
class passrate: 0.15 (3/20)
total ruff issues: 45
total mypy errors: 21
Saved test summary to results_summary.json | Passrate: 3/20 (0.15%)
```

Total syntax errors refer to the number of test cases that could not be evaluated due to syntax errors in the generated code.
The pass@1, total ruff issues, total mypy errors is testcase passrate for cases without syntax errors.
