# OREO setup

This document is the detailed install guide for the public OREO Dual-Stage Contrastive FlowEdit release (Python/CUDA env, reference code, model and dataset downloads, offline mode). For a short product overview plus train/eval commands, see [`README.md`](README.md). The workflow is standalone Linux/CUDA and does not require Koala, a private S3 bucket, or internal infrastructure.

## Requirements

The verified environment uses:

| Component | Version |
| --- | --- |
| Python | 3.12 |
| CUDA | 12.4 |
| PyTorch | 2.6.0+cu124 |
| Torchvision | 0.21.0+cu124 |
| Diffusers | 0.38.0 |
| Transformers | 4.57.6 |
| Tokenizers | 0.22.0 |
| Accelerate | 1.4.0 |
| Safetensors | 0.8.0 |
| Flash Attention | 2.7.3 |
| Spconv | 2.3.8 |
| PEFT | 0.17.1 |
| Open3D | 0.19.0 |

Use Linux with an NVIDIA GPU and a CUDA 12.4 compatible driver (enough VRAM for TRELLIS + Qwen-Image-Edit in shared-device mode; one GPU is fine for smoke tests). The remaining Python dependencies are pinned in [`requirements.txt`](requirements.txt), including `kaolin==0.18.0` from the matching torch 2.6.0 / cu124 wheel index. Do not upgrade Transformers to 5.x (breaks Qwen prompt encoding with Diffusers 0.38.0). Build Flash Attention after installing the matching PyTorch wheel, and set `TORCH_CUDA_ARCH_LIST` to arches your GPUs actually need.

## Install the environment

```bash
conda create -n oreo python=3.12 -y
conda activate oreo

pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 \
  --index-url https://download.pytorch.org/whl/cu124
pip install spconv-cu124==2.3.8
pip install kaolin==0.18.0 \
  -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.6.0_cu124.html
pip install peft==0.17.1

export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
pip install --no-build-isolation flash-attn==2.7.3
pip install git+https://github.com/NVlabs/nvdiffrast.git@v0.3.3
pip install --no-build-isolation \
  "git+https://github.com/autonomousvision/mip-splatting.git#subdirectory=submodules/diff-gaussian-rasterization"
pip install -r requirements.txt
```

Install the reference implementations required by the TRELLIS adapter:

```bash
mkdir -p _reference_codes
git clone --recursive https://github.com/microsoft/TRELLIS.git \
  _reference_codes/TRELLIS
git clone https://github.com/VAST-AI-Research/TripoSF.git \
  _reference_codes/TripoSF
```

Set the runtime variables before training or evaluation:

```bash
export ATTN_BACKEND=flash_attn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SPCONV_ALGO=auto
export CUDA_VISIBLE_DEVICES=0   # or a comma-separated GPU list
```

Set `HF_HOME` and `TORCH_HOME` if the model caches should live outside their default locations. Set `HF_TOKEN` only when Hugging Face requires authentication for a download.

## Download model assets

The retained training configuration expects TRELLIS weights at `pretrained_weights/TRELLIS-image-large` and uses the public Qwen-Image-Edit-2511 model through the Hugging Face cache.

```bash
mkdir -p pretrained_weights
python scripts/download/download_trellis.py

python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    "Qwen/Qwen-Image-Edit-2511",
    revision="6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9",
)
PY
```

Download the local evaluation assets when their metrics are needed:

```bash
python scripts/download/download_clip.py
python scripts/download/download_dinov2.py --mode hf \
  --repo facebook/dinov2-giant \
  --dest pretrained_weights/dinov2-giant
```

CLIP and DINOv2 are used by the canonical evaluator. Both checkpoints are publicly downloadable.

## Download datasets

AlphaImages v3 is the default training and evaluation dataset:

```bash
python scripts/download/download_alphaimages_v3.py \
  --out dataset/alphaimages_v3 \
  --token "${HF_TOKEN:-}"
```

GSO and Toys4K are optional evaluation datasets:

```bash
bash scripts/download/prepare_gso_dataset.sh
bash scripts/download/prepare_toys4k_dataset.sh
```

The prepared datasets are consumed by the training and evaluation entry points.

## Released OREO checkpoint (`oreo_v1`)

Public weights live on Hugging Face at [`TencentARC/OREO`](https://huggingface.co/TencentARC/OREO) (`oreo_v1`, from training `checkpoint_10_3157`). The Dual-Stage loader expects a **flat** directory with both:

- `model.safetensors` → dense / `slat_flow_model`
- `model_1.safetensors` → sparse / `sparse_structure_flow_model`

```bash
hf download TencentARC/OREO --include 'oreo_v1/*' --local-dir ./checkpoints/oreo_v1
# Confirm the two .safetensors files sit directly under ./checkpoints/oreo_v1/
```

## Train and eval entry points

From the repository root (after the environment, `_reference_codes/`, weights, and datasets are ready):

```bash
# Train — single node / few GPUs, or multi-GPU launcher
CUDA_VISIBLE_DEVICES=0 bash scripts/single_node/main_trellis_oreo.sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/multi_node/main_trellis_oreo.sh

# Eval — CKPT_INPUT may be empty (pretrained baseline), a flat Dual-Stage dir, or a checkpoints/ folder
bash scripts/eval/eval_trellis_dense_flowedit_DDP.sh ./checkpoints/oreo_v1 alphaimages
```

`EVAL_DATASET` options include `alphaimages`, `alphaimage_train`, `gso_selected`, and `toys4k`. See [`README.md`](README.md) for the short command list.

## Offline use

After the model and dataset downloads complete, set the cache locations and enable offline mode:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
```

Keep the following paths available from the repository root:

```text
_reference_codes/TRELLIS
_reference_codes/TripoSF
pretrained_weights/TRELLIS-image-large
pretrained_weights/dinov2-giant
dataset/alphaimages_v3/train
dataset/alphaimages_v3/test
checkpoints/oreo_v1/model.safetensors
checkpoints/oreo_v1/model_1.safetensors
```

If imports fail early with missing TRELLIS modules, re-check the `_reference_codes/TRELLIS` clone (`--recursive`). If checkpoint load raises `FileNotFoundError`, both flat safetensors files must be present in the checkpoint directory.
