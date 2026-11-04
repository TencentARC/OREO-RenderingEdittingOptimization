#!/bin/bash
# Prepare GSO (Google Scanned Objects) eval set.
#
# Steps:
#   1) HF download Roldbach/google_scanned_objects (~4.7GB zip)
#   2) Unzip -> sample ~100 objects -> 512x512 white-bg front views
#   3) Symlink dataset/gso_test in the repo
#
# Usage:
#   bash scripts/download/prepare_gso_dataset.sh
#   bash scripts/download/prepare_gso_dataset.sh --inspect
#   bash scripts/download/prepare_gso_dataset.sh --local_zip /path/to.zip
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_ROOT"

source /home/zhiyuan_ma/anaconda3/etc/profile.d/conda.sh
conda activate grpo3d_trellis

echo "========================================"
echo "GSO Dataset preparation"
echo "========================================"
echo "Project root directory: $REPO_ROOT"
echo "========================================"

PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}" \
python scripts/download/prepare_gso_dataset.py "$@"
