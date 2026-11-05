#!/bin/bash
# Teacher/Student TRELLIS evaluation launcher.
#
# Thin wrapper around the DDP entrypoint. Defaults to oreo_v1 + alphaimages.
#
# Usage:
#   bash scripts/eval/eval_trellis.sh
#   bash scripts/eval/eval_trellis.sh ./checkpoints/oreo_v1
#   bash scripts/eval/eval_trellis.sh ./checkpoints/oreo_v1 alphaimages
#   bash scripts/eval/eval_trellis.sh ./checkpoints/oreo_v1 alphaimages --config.data.eval.n_view=5
#
set -euo pipefail

CKPT_INPUT="${1:-./checkpoints/oreo_v1}"
EVAL_DATASET="${2:-alphaimages}"
if [ "$#" -ge 2 ]; then
    shift 2
elif [ "$#" -eq 1 ]; then
    shift
fi

exec bash scripts/eval/eval_trellis_DDP.sh "$CKPT_INPUT" "$EVAL_DATASET" "$@"
