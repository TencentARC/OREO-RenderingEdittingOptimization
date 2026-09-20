#!/bin/bash
# TRELLIS Dual-Stage (Sparse + Dense) Contrastive FlowEdit Training script (stand-alone version)
#
# GPU Allocation strategy (shared mode):
# - All N cards used simultaneously Trellis training (DDP) and Guidance (FlowEdit)
#
# Usage example:
# - Single card training:export CUDA_VISIBLE_DEVICES=0
# - 2Card training:export CUDA_VISIBLE_DEVICES=0,1

# Calculate the number of training cards
: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GPU_COUNT=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
TRAIN_GPU_COUNT=$GPU_COUNT
RUN_NAME="${RUN_NAME:-trellis_oreo}"

echo "========================================"
echo "GPU allocation information"
echo "========================================"
echo "visible GPU: $CUDA_VISIBLE_DEVICES ($GPU_COUNT Zhang)"
echo "Number of training processes: $TRAIN_GPU_COUNT"
echo "training + Guidance: cuda:0-$((GPU_COUNT-1))(share the same device)"
echo "========================================"

python -m accelerate.commands.launch \
    --num_processes=$TRAIN_GPU_COUNT \
    -m oreo.systems.trellis.entries.oreo_autograd \
    --config=config/trellis_oreo.py \
    --config.eval_only=false \
    --config.run_name="$RUN_NAME"
