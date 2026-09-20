# OREO setup

This document is the setup guide for the public OREO Dual-Stage Contrastive FlowEdit release. It describes a standalone Linux/CUDA environment and does not require Koala, a private S3 bucket, or internal infrastructure.

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

Use Linux with an NVIDIA GPU and a CUDA 12.4 compatible driver. The remaining Python dependencies are pinned in [`requirements.txt`](requirements.txt). Do not upgrade Transformers to 5.x in this environment. Build Flash Attention after installing the matching PyTorch wheel.

## Install the environment

```bash
conda create -n oreo python=3.12 -y
conda activate oreo

pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 \
  --index-url https://download.pytorch.org/whl/cu124
pip install spconv-cu124==2.3.8
pip install kaolin \
  -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.6.0_cu124.html

export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
pip install --no-build-isolation flash-attn==2.7.3
pip install git+https://github.com/NVlabs/nvdiffrast.git@v0.3.3
pip install -r requirements.txt
```

Set the runtime variables before training or evaluation:

```bash
export ATTN_BACKEND=flash_attn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
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
  --dest pretrained_weights/dinov2-giant
```

CLIP is used by the canonical evaluator. DINOv2 is used by training. The DINO metric uses the public DINOv3 ViT-L/16 checkpoint:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    "facebook/dinov3-vitl16-pretrain-lvd1689m",
    local_dir="pretrained_weights/dinov3-vitl16-pretrain-lvd1689m/facebook/dinov3-vitl16-pretrain-lvd1689m",
)
PY
```

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

The dense evaluator accepts `alphaimages`, `alphaimage_train`, `gso_selected`, and `toys4k` through its DDP wrapper.

## Training

For a single GPU or a small local run:

```bash
CUDA_VISIBLE_DEVICES=0 \
bash scripts/single_node/main_trellis_dualstage_contrastive.sh
```

For a multi-GPU run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash scripts/multi_node/main_trellis_dualstage_contrast.sh
```

Both launchers run `edit4shape.systems.trellis.entries.contrastive_dualstage_autograd` with `config/trellis_stage1+2_contrastive.py`. Set `RUN_NAME` or pass configuration overrides to customize a run.

## Evaluation

The canonical evaluator compares the pretrained teacher path with the finetuned student path using Dense FlowEdit and CLIP/DINO metrics.

```bash
bash scripts/eval/eval_trellis_dense_flowedit_DDP.sh \
  /path/to/checkpoints/checkpoint_10_3157 alphaimages
```

The wrapper also accepts `alphaimage_train`, `gso_selected`, and `toys4k`. FlowEdit parameters can be passed after the dataset name:

```bash
bash scripts/eval/eval_trellis_dense_flowedit_DDP.sh \
  /path/to/checkpoints/checkpoint_10_3157 gso_selected \
  --config.rollout.flowedit.n_max=12 \
  --config.rollout.flowedit.cfg_scale_tgt=4.0 \
  --config.rollout.flowedit.cfg_scale_src=-4.0
```

For a custom process layout, launch [`scripts/eval/eval_trellis_dense_flowedit.py`](scripts/eval/eval_trellis_dense_flowedit.py) directly with `accelerate`.

## Checkpoint export

```bash
bash scripts/export/export_checkpoint.sh \
  /path/to/checkpoint \
  /path/to/output
```

The Python implementation is available at [`scripts/export/export_checkpoint.py`](scripts/export/export_checkpoint.py).

## Offline use

After the model and dataset downloads complete, set the cache locations and enable offline mode:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
```

Keep the following paths available from the repository root:

```text
pretrained_weights/TRELLIS-image-large
dataset/alphaimages_v3/train
dataset/alphaimages_v3/test
```
