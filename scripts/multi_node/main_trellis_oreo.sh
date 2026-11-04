#!/bin/bash
# Multi-node TRELLIS Dual-Stage (Sparse + Dense) Contrastive FlowEdit training.
#
# Training:
#   Sparse: Pretrained Rollout -> Noise -> Student velocity -> Contrastive loss
#   Dense:  Reuse Sparse c_src/c_tgt -> Dense velocity -> Contrastive loss
#
# GPU policy (shared): all N cards for Trellis DDP + FlowEdit Guidance.
#
# Examples:
# - Default 8 GPUs: CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./main_trellis_oreo.sh
# - Custom: CUDA_VISIBLE_DEVICES=4,5,6,7 ./main_trellis_oreo.sh
#
: "${CUDA_VISIBLE_DEVICES:=0,1,2,3,4,5,6,7}"
RUN_NAME="${RUN_NAME:-trellis_oreo}"
: "${MASTER_PORT:=29512}"

export CUDA_VISIBLE_DEVICES
GPU_COUNT=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

TRAIN_GPU_COUNT=$GPU_COUNT

echo "========================================"
echo "GPU allocation information (Dual-Stage Contrastive FlowEdit — Latent spatial contrastive learning)"
echo "========================================"
echo "visible GPU: $CUDA_VISIBLE_DEVICES ($GPU_COUNT Zhang)"
echo "Number of training processes: $TRAIN_GPU_COUNT"
echo "training + Guidance: cuda:0-$((GPU_COUNT-1))(share the same device)"
echo "Rendering mode: GS Color"
echo "========================================"

python -m accelerate.commands.launch \
  --num_processes=${TRAIN_GPU_COUNT} \
  --multi_gpu \
  --main_process_port=${MASTER_PORT} \
  -m oreo.systems.trellis.entries.oreo_autograd \
  --config=config/trellis_oreo.py \
  --config.eval_only=False \
  --config.use_wandb=True \
  --config.run_name="$RUN_NAME" \
  "$@"
