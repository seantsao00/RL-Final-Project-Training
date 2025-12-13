WANDB_ENTITY=wcwutw-ntu
WANDB_PROJECT=rl-final-classeval-compositional

srun -N 1 -n 1 --gpus-per-node 1 -A ACD114118 \
uv run -m src.train_grpo_lora_classeval \
  --config configs/3b_lora_classeval.yaml
