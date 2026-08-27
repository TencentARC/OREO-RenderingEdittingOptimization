#!/usr/bin/env bash
set -euo pipefail

# Dedicated formal variant of run_8gpu_teacher_cfg_true_mp_no.sh.  The shared
# launcher owns every other run setting; this immutable wrapper selects the
# derived config that changes only training camera yaw from [0,360] to
# [90,270].
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_8gpu_teacher_cfg_mp_no.sh" true \
    "--config=config/trellis_stage1+2_contrastive_yaw90_270.py" "$@"
