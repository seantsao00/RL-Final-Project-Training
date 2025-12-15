#!/bin/bash
# Evaluation script for ClassEval holistic model

# Default values
MODEL_PATH="${1:-./checkpoints/grpo-classeval-holistic}"
OUTPUT_DIR="${2:-./evaluation_results}"
MAX_EXAMPLES="${3:-}"
TEMPERATURE="${4:-0.7}"

# Build command
CMD="python -m src.classeval_holistic_evaluate \
    --model_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --temperature $TEMPERATURE \
    --max_samples 1000 \
    --train_test_split_ratio 0.8 \
    --tests_weight 1.0 \
    --ruff_weight 0.5 \
    --mypy_weight 1.5"

# Add max_examples if specified
if [ -n "$MAX_EXAMPLES" ]; then
    CMD="$CMD --max_examples $MAX_EXAMPLES"
fi

echo "Running evaluation with:"
echo "  Model: $MODEL_PATH"
echo "  Output: $OUTPUT_DIR"
echo "  Temperature: $TEMPERATURE"
if [ -n "$MAX_EXAMPLES" ]; then
    echo "  Max examples: $MAX_EXAMPLES"
fi
echo ""

# Run evaluation
eval $CMD
