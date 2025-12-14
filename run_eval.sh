srun -N 1 -n 1 --gpus-per-node 1 -A ACD114118 \
uv run -m classeval_eval.generate \
  --test_work \
  --config configs/3b_lora_classeval.yaml
