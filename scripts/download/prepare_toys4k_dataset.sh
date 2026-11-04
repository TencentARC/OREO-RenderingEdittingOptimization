#!/bin/bash
# Prepare Toys4k single-view test set (GSO-aligned).
#
# Steps:
#   1) Extract .blend from toys4k_blend_files.zip
#   2) Blender batch render (elev 30deg, azim 0deg, 512x512 RGBA)
#   3) Symlink dataset/toys4k_test
#
# Usage:
#   bash scripts/download/prepare_toys4k_dataset.sh
#   bash scripts/download/prepare_toys4k_dataset.sh --max_objects 10
#   bash scripts/download/prepare_toys4k_dataset.sh --resume
#   bash scripts/download/prepare_toys4k_dataset.sh --skip_extract --resume
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_ROOT"

source /home/zhiyuan_ma/anaconda3/etc/profile.d/conda.sh
conda activate grpo3d_trellis

echo "========================================"
echo "Toys4k Dataset preparation"
echo "========================================"
echo "Project root directory: $REPO_ROOT"
echo "========================================"

PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}" \
python scripts/download/prepare_toys4k_dataset.py \
    --workers 4 \
    "$@"
