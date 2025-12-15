# Model Evaluation Guide

This document describes how to evaluate both the trained GRPO model and the baseline model on the ClassEval holistic dataset.

## Evaluation Scripts

### Trained Model Evaluation
The evaluation script (`src/classeval_holistic_evaluate.py`) loads a trained model and evaluates it on the test split of the ClassEval dataset.

### Baseline Model Evaluation
The baseline evaluation script (`src/evaluate_baseline.py`) evaluates the original Qwen2.5-Coder-3B-Instruct model (before training) for comparison purposes.

### Metrics Collected

For each generated solution, the script computes:

1. **Unit Test Pass Rate**: Percentage of unit tests passed
2. **Ruff Score**: Code quality score (0-1)
3. **Mypy Score**: Type checking score (0-1)
4. **Overall Score**: Weighted average of all metrics
5. **Syntax Errors**: Whether the generated code has syntax errors

### Aggregated Metrics

Across all examples:
- Average test pass rate
- Average ruff/mypy scores
- Average overall score
- Syntax error rate
- Perfect solution rate (100% tests passed)

## Usage

### Evaluating Trained Model

#### Using the Bash Script (Recommended)

```bash
# Evaluate the default trained model
./evaluate.sh

# Evaluate a specific checkpoint
./evaluate.sh ./checkpoints/checkpoint-100 ./evaluation_results 50 0.7
#             [model_path]                  [output_dir]        [max_examples] [temperature]
#### Using Python Directly

```bash
# Basic evaluation
python -m src.classeval_holistic_evaluate \
    --model_path ./checkpoints/grpo-classeval-holistic \
    --output_dir ./evaluation_results

# Evaluate with specific parameters
python -m src.classeval_holistic_evaluate \
    --model_path ./checkpoints/checkpoint-100 \
    --output_dir ./evaluation_results \
    --max_examples 50 \
    --temperature 0.7 \
    --max_new_tokens 1024 \
    --tests_weight 1.0 \
    --ruff_weight 0.5 \
    --mypy_weight 1.5
```

### Evaluating Baseline Model

## Command-Line Arguments

### Trained Model Evaluation (`classeval_holistic_evaluate.py`)

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_path` | Required | Path to trained model directory |
| `--output_dir` | `./evaluation_results` | Directory to save results |
| `--max_examples` | None | Max examples to evaluate (None = all) |
| `--max_samples` | 1000 | Max samples from dataset before split |
| `--train_test_split_ratio` | 0.8 | Train/test split ratio |
| `--max_new_tokens` | 1024 | Maximum tokens to generate |
| `--temperature` | 0.7 | Sampling temperature |
| `--tests_weight` | 1.0 | Weight for unit test reward |
| `--ruff_weight` | 0.5 | Weight for ruff reward |
| `--mypy_weight` | 1.5 | Weight for mypy reward |

### Baseline Model Evaluation (`evaluate_baseline.py`)

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_name` | `Qwen/Qwen2.5-Coder-3B-Instruct` | HuggingFace model name |
| `--output_dir` | `./evaluation_results` | Directory to save results |
| `--max_examples` | None | Max examples to evaluate (None = all) |
| `--max_samples` | 1000 | Max samples from dataset before split |
| `--train_test_split_ratio` | 0.8 | Train/test split ratio |
| `--max_new_tokens` | 1024 | Maximum tokens to generate |
| `--temperature` | 0.7 | Sampling temperature |
| `--tests_weight` | 1.0 | Weight for unit test reward |
| `--ruff_weight` | 0.5 | Weight for ruff reward |
| `--mypy_weight` | 1.5 | Weight for mypy reward |
#### Using Python Directly

```bash
# Basic baseline evaluation
python -m src.evaluate_baseline \
    --model_name Qwen/Qwen2.5-Coder-3B-Instruct \
    --output_dir ./evaluation_results

# Evaluate with specific parameters
python -m src.evaluate_baseline \
    --model_name Qwen/Qwen2.5-Coder-3B-Instruct \
    --output_dir ./evaluation_results \
    --max_examples 50 \
    --temperature 0.7
``` --ruff_weight 0.5 \
    --mypy_weight 1.5
```

## Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_path` | Required | Path to trained model directory |
| `--output_dir` | `./evaluation_results` | Directory to save results |
| `--max_examples` | None | Max examples to evaluate (None = all) |
| `--max_samples` | 1000 | Max samples from dataset before split |
| `--train_test_split_ratio` | 0.8 | Train/test split ratio |
| `--max_new_tokens` | 1024 | Maximum tokens to generate |
| `--temperature` | 0.7 | Sampling temperature |
| `--tests_weight` | 1.0 | Weight for unit test reward |
| `--ruff_weight` | 0.5 | Weight for ruff reward |
| `--mypy_weight` | 1.5 | Weight for mypy reward |

## Output

### Console Output

The script prints:
## Examples

### Quick Test (10 examples)

```bash
# Test trained model
python -m src.classeval_holistic_evaluate \
    --model_path ./checkpoints/grpo-classeval-holistic \
    --max_examples 10 \
    --temperature 0.5

# Test baseline model
python -m src.evaluate_baseline \
    --max_examples 10 \
    --temperature 0.5
```

### Full Evaluation with Greedy Decoding

```bash
# Trained model
python -m src.classeval_holistic_evaluate \
    --model_path ./checkpoints/grpo-classeval-holistic \
    --temperature 0.0 \
    --output_dir ./evaluation_results/greedy

# Baseline model
python -m src.evaluate_baseline \
    --temperature 0.0 \
    --output_dir ./evaluation_results/baseline_greedy
```

### Compare Trained vs Baseline

```bash
# Evaluate baseline
./evaluate_baseline.sh "Qwen/Qwen2.5-Coder-3B-Instruct" ./evaluation_results 100 0.7

# Evaluate trained model
./evaluate.sh ./checkpoints/grpo-classeval-holistic ./evaluation_results 100 0.7

# Results will be saved to:
# - evaluation_results/Qwen_Qwen2.5-Coder-3B-Instruct_baseline_evaluation.json
# - evaluation_results/grpo-classeval-holistic_evaluation.json
```

### Compare Multiple Checkpoints

```bash
# Evaluate baseline first
./evaluate_baseline.sh

# Then evaluate all checkpoints
for checkpoint in ./checkpoints/checkpoint-*; do
    echo "Evaluating $checkpoint"
    ./evaluate.sh "$checkpoint" ./evaluation_results 50 0.7
done
``` "avg_test_pass_rate": 0.725,
    "avg_ruff_score": 0.8234,
    ...
  },
  "detailed_results": [
    {
      "example_id": 0,
      "prompt": [...],
      "completion": "...",
      "tests": "...",
      "metrics": {
        "test_passed": 8,
        "test_total": 10,
        "test_pass_rate": 0.8,
        ...
      }
    },
    ...
  ]
}
```

## Examples

### Quick Test (10 examples)

```bash
python -m src.classeval_holistic_evaluate \
    --model_path ./checkpoints/grpo-classeval-holistic \
    --max_examples 10 \
    --temperature 0.5
```

### Full Evaluation with Greedy Decoding

```bash
python -m src.classeval_holistic_evaluate \
    --model_path ./checkpoints/grpo-classeval-holistic \
    --temperature 0.0 \
    --output_dir ./evaluation_results/greedy
```

### Compare Multiple Checkpoints

```bash
for checkpoint in ./checkpoints/checkpoint-*; do
    echo "Evaluating $checkpoint"
    ./evaluate.sh "$checkpoint" ./evaluation_results 50 0.7
done
```

## Notes

- The script uses the same test split configuration as training (80/20 split)
- GPU is automatically used if available
- Evaluation uses `bfloat16` precision on CUDA for efficiency
- Each example is evaluated independently (no batching for accuracy)
- The same reward weights from training are used by default
