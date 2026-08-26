#!/usr/bin/env bash
set -euo pipefail

exec bash "$(dirname "${BASH_SOURCE[0]}")/run_8gpu_teacher_cfg_mp_no.sh" false "$@"
