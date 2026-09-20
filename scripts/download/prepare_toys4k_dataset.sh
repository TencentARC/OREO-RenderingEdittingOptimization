#!/bin/bash
# Prepare Toys4k Single view test data set (aligned GSO format)
#
# Function:
#   1. downloaded from toys4k_blend_files.zip medium decompression .blend File
#   2. use Blender Batch rendering of single views (elevation angle 30°, azimuth angle 0°,512×512 RGBA)
#   3. output toys4k_0000.png ... Arrive /data/zhiyuan_ma/data/toys4k_test/
#   4. in dataset/toys4k_test Create soft link
#
# Usage:
#   conda activate grpo3d_trellis
#   bash scripts/download/prepare_toys4k_dataset.sh
#
#   # Debugging: only before rendering 10 a
#   bash scripts/download/prepare_toys4k_dataset.sh --max_objects 10
#
#   # Resume upload from breakpoint (skip rendered)
#   bash scripts/download/prepare_toys4k_dataset.sh --resume
#
#   # Skip decompression if already decompressed
#   bash scripts/download/prepare_toys4k_dataset.sh --skip_extract --resume
#
# Estimated time (4 workers,RTX3090 GPU):
#   - Unzipped: approx. 5 minutes
#   - rendering 4000 objects: approx. 2–4 hours
#   - Total: approx. 2–5 hours

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
