uv run -m classeval_eval.generate \
  --config configs/3b_lora_classeval.yaml \
  --eval_output_dir classeval_eval/results/our_eval_output \
  --greedy 0 \
  --lora_weights checkpoints/checkpoint-105
  # --test_work 1 \
