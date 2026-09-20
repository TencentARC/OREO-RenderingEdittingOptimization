#!/bin/bash
# Prepare GSO (Google Scanned Objects) test data set
#
# Function:
#   1. from HuggingFace Download Roldbach/google_scanned_objects(single zip,~4.7 GB)
#   2. Unzip to /data/zhiyuan_ma/data/gso_extracted/
#   3. sampling 100 object, extract the front view reference image (512×512 white background RGB)
#   4. Output to /data/zhiyuan_ma/data/gso_test/
#   5. Create in code directory dataset/gso_test soft link
#
# Usage:
#   # Full run (download + Unzip + extraction)
#   conda activate grpo3d_trellis
#   bash scripts/download/prepare_gso_dataset.sh
#
#   # Check only zip Internal structure (do not extract images)
#   bash scripts/download/prepare_gso_dataset.sh --inspect
#
#   # Already zip Skip download when
#   bash scripts/download/prepare_gso_dataset.sh --local_zip /data/zhiyuan_ma/data/gso/google_scanned_objects.zip
#
# Estimated time:
#   - Download: approx. 1–2 minutes (~4.7 GB, depends on the Internet speed; campus online appointment 30–60 s)
#   - Unzipped: approx. 3–5 minutes (zip Contains 54590 files)
#   - Image extraction: approx. 1 minutes
#   - Total: approx. 5–10 minutes

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
