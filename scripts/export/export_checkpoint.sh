#!/bin/bash
# Export a training checkpoint to a TRELLIS-inference-compatible weight directory.
#
# Usage:
#   bash scripts/export/export_checkpoint.sh <checkpoint_path> <output_path> [pretrained]
#
set -euo pipefail

CHECKPOINT="${1:?Usage: $0 <checkpoint_path> <output_path>}"
OUTPUT="${2:?Usage: $0 <checkpoint_path> <output_path>}"
PRETRAINED="${3:-pretrained_weights/TRELLIS-image-large}"

echo "========================================"
echo "Export Checkpoint → TRELLIS reasoning format"
echo "========================================"
echo "Checkpoint:  $CHECKPOINT"
echo "Output:      $OUTPUT"
echo "Pretrained:  $PRETRAINED"
echo "========================================"

python scripts/export/export_checkpoint.py \
    --checkpoint "$CHECKPOINT" \
    --output "$OUTPUT" \
    --pretrained "$PRETRAINED"
