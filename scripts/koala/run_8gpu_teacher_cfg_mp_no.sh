#!/usr/bin/env bash
set -euo pipefail

# Common entry point for the two authorized teacher-CFG A/B runs.  The two
# public wrappers in this directory pass the only training-variable value;
# this file deliberately contains no source-rewriting path.

TEACHER_CFG="${1:-}"
shift || true
case "${TEACHER_CFG}" in
    true)  TEACHER_CFG_ABSL=True ;;
    false) TEACHER_CFG_ABSL=False ;;
    *)
        echo "ERROR: expected teacher_cfg true or false, got: ${TEACHER_CFG}" >&2
        exit 2
        ;;
esac

cd /data/work/run_codes

# The fast setup restores only the pinned, immutable environment and assets.
# It also starts a generic log sync; replace that sync with this run's isolated
# output sync below so A/B outputs cannot overwrite another experiment.
. scripts/setup_koala.sh --fast

export PATH="/tmp/uv-venv/bin:${PATH}"
export HF_HOME=/local-ssd/hf_cache
export HF_HUB_DISABLE_XET=1
export TORCH_HOME=/local-ssd/torch_home
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export UV_PROJECT_ENVIRONMENT=/tmp/uv-venv
export UV_FROZEN=1
export ATTN_BACKEND=flash_attn
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

SOURCE_COMMIT="${SOURCE_COMMIT:?SOURCE_COMMIT must be supplied by the immutable submission command}"
CODE_BUNDLE_URI="${CODE_BUNDLE_URI:?CODE_BUNDLE_URI must be supplied by the immutable submission command}"
RUN_NAME="${RUN_NAME:?RUN_NAME must be supplied by the immutable submission command}"
QWEN_COMMIT="6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9"
QWEN_ARCHIVE_URI="s3://arcwm-code-us-west-2/ericzyma/data/flow_grpo_v2/models/qwen-image-edit-2511-${QWEN_COMMIT}-hf-v1.tar"
OUTPUT_URI="${OUTPUT_URI:-s3://arcwm-code-us-west-2/ericzyma/oreo-training/${RUN_NAME}/}"
RUN_DIR="/data/work/run_codes/logs/${RUN_NAME}"
STATUS_DIR="/local-ssd/${RUN_NAME}-status"

case "${RUN_NAME}" in
    *tc-${TEACHER_CFG}-mp-no*) ;;
    *)
        echo "ERROR: RUN_NAME must contain tc-${TEACHER_CFG}-mp-no: ${RUN_NAME}" >&2
        exit 2
        ;;
esac

# setup_koala.sh's broad sync is not part of this experiment's output path.
if [ -n "${SYNC_PID:-}" ]; then
    kill "${SYNC_PID}" 2>/dev/null || true
fi

mkdir -p "${STATUS_DIR}"

sync_run() {
    if [ -d "${RUN_DIR}" ]; then
        s5cmd sync "${RUN_DIR}/" "${OUTPUT_URI}run/" || true
    fi
    s5cmd sync "${STATUS_DIR}/" "${OUTPUT_URI}status/" || true
}

RUN_SYNC_PID=""
cleanup() {
    if [ -n "${RUN_SYNC_PID}" ]; then
        kill "${RUN_SYNC_PID}" 2>/dev/null || true
    fi
    sync_run
}
trap cleanup EXIT

cat > "${STATUS_DIR}/provenance.env" <<EOF
source_commit=${SOURCE_COMMIT}
code_bundle_uri=${CODE_BUNDLE_URI}
qwen_commit=${QWEN_COMMIT}
qwen_archive=${QWEN_ARCHIVE_URI}
qwen_model_path=${QWEN_IMAGE_EDIT_MODEL_PATH}
source_guidance=reverse_cfg
target_cfg_scale=4
source_cfg_scale=-4
target_prompt=Rotate the camera. White background.
source_prompt=Rotate the camera. White background.
teacher_cfg=${TEACHER_CFG}
mixed_precision=no
checkpoint=
student_denoise_cfg=false
seed=42
train_batch_size=1
resolution=1024
flowedit_steps=12
flowedit_n_max=9
optimizer=adan
optimizer_lr=1e-4
wandb_project=trellis-dual-contrastive
wandb_run_name=${RUN_NAME}
world_size=8
EOF

printf '%s\n' \
    "{\"status\":\"STARTING\",\"source_commit\":\"${SOURCE_COMMIT}\",\"code_bundle_uri\":\"${CODE_BUNDLE_URI}\",\"qwen_commit\":\"${QWEN_COMMIT}\",\"teacher_cfg\":${TEACHER_CFG},\"mixed_precision\":\"no\",\"source_cfg_scale\":-4,\"target_cfg_scale\":4,\"student_denoise_cfg\":false,\"flowedit_steps\":12,\"flowedit_n_max\":9,\"world_size\":8}" \
    > "${STATUS_DIR}/status.json"
sync_run

(while true; do sleep 300; sync_run; done) &
RUN_SYNC_PID=$!

# Immutable runtime/model preflight.  These checks are provider-free and must
# complete before the training entry point is started.
python scripts/qwen_snapshot_archive.py verify-cache \
    "${HF_HOME}/hub/models--Qwen--Qwen-Image-Edit-2511"
python scripts/test_qwen_environment.py

training_status="PASS"
if python -m accelerate.commands.launch \
    --num_processes=8 \
    --multi_gpu \
    --main_process_port=29512 \
    -m edit4shape.systems.trellis.entries.contrastive_dualstage_autograd \
    --config=config/trellis_stage1+2_contrastive.py \
    --config.eval_only=False \
    --config.use_wandb=True \
    --config.run_name="${RUN_NAME}" \
    --config.logdir=logs \
    --config.num_epochs=500 \
    --config.checkpoint="" \
    --config.mixed_precision=no \
    --config.train.loss.contrastive.teacher_cfg="${TEACHER_CFG_ABSL}" \
    --config.train.student_denoise_cfg=False \
    --config.guidance.flowedit.steps=12 \
    --config.guidance.flowedit.n_max=9 \
    --config.train.guidance.true_cfg_scale_src=-4 \
    --config.train.guidance.true_cfg_scale_tgt=4 \
    "--config.train.guidance.target_prompt=Rotate the camera. White background." \
    "--config.train.guidance.source_prompt=Rotate the camera. White background." \
    --config.guidance.model_path="${QWEN_IMAGE_EDIT_MODEL_PATH}"; then
    :
else
    training_status="FAILED"
fi

printf '%s\n' \
    "{\"status\":\"${training_status}\",\"source_commit\":\"${SOURCE_COMMIT}\",\"code_bundle_uri\":\"${CODE_BUNDLE_URI}\",\"qwen_commit\":\"${QWEN_COMMIT}\",\"teacher_cfg\":${TEACHER_CFG},\"mixed_precision\":\"no\",\"source_cfg_scale\":-4,\"target_cfg_scale\":4,\"student_denoise_cfg\":false,\"flowedit_steps\":12,\"flowedit_n_max\":9,\"world_size\":8}" \
    > "${STATUS_DIR}/status.json"
if [ "${training_status}" = "FAILED" ]; then
    exit 1
fi
