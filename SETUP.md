# OREO setup

This document is the source of truth for the OREO Dual-Stage Contrastive FlowEdit environment, model assets, data preparation, and run commands.

## Verified environment

The release targets Python 3.12 and CUDA 12.4 with the following core versions:

| Component | Version |
| --- | --- |
| Python | 3.12 |
| PyTorch | 2.6.0+cu124 |
| Torchvision | 0.21.0+cu124 |
| Diffusers | 0.38.0 |
| Transformers | 4.57.6 |
| Tokenizers | 0.22.0 |
| Accelerate | 1.4.0 |
| Safetensors | 0.8.0 |
| Flash Attention | 2.7.3 |
| Spconv | 2.3.8 |

The complete Python dependency set is pinned in [`requirements.txt`](requirements.txt). Do not upgrade Transformers to 5.x in this environment. Flash Attention must be built after PyTorch 2.6.0+cu124 is installed.

## Koala setup

The recommended path is the project setup script on a Koala pod. The script must be sourced so that its environment variables remain in the current shell.

Prerequisites for the Koala path are `koala`, `s5cmd`, `aws`, `uv`, access to the project S3 bucket, and an `HF_TOKEN` for first-time Hugging Face downloads. `WANDB_API_KEY` is optional and is only needed for experiment logging.

```bash
# First-time setup: install or restore the environment and download assets.
. scripts/setup_koala.sh --download

# Daily setup: restore only from the pinned offline caches.
. scripts/setup_koala.sh --fast

# Restore only the environment and Qwen Image Edit cache.
. scripts/setup_koala.sh --fast --qwen-only
```

The `--fast` path is intentionally offline. It fails if a required immutable cache is missing instead of falling back to an older environment or contacting Hugging Face. The setup script also creates these project links when the corresponding assets are available:

```text
pretrained_weights -> /local-ssd/pretrained_weights
dataset/alphaimages_v3 -> /local-ssd/alphaimages_v3
```

## Required model and data assets

`setup_koala.sh --download` prepares the assets used by the retained training and evaluation paths:

- TRELLIS-image-large weights;
- Qwen-Image-Edit-2511 at commit `6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9`;
- AlphaImages v3 training and test data;
- the DINOv2 torch hub cache used by training.

The setup script stores and restores these assets through the immutable cache layout configured in the script. Do not replace the pinned Qwen archive with the legacy archive.

For optional local preparation, the following utilities remain available:

```bash
python scripts/download/download_birefnet.py
python scripts/download/download_dinov2.py --out pretrained_weights/dinov2-giant
```

RMBG-2.0 is restored by the Koala setup script when its cache is available. CLIP and DINOv3-L caches are optional evaluation assets; missing caches disable the corresponding metrics.

## Additional evaluation datasets

AlphaImages v3 is prepared by the Koala setup script. GSO and Toys4K are optional evaluation datasets:

```bash
bash scripts/download/prepare_gso_dataset.sh
bash scripts/download/prepare_toys4k_dataset.sh
```

The dense evaluator accepts `alphaimages`, `alphaimage_train`, `gso_selected`, and `toys4k` through the DDP wrapper.

## Training

For a local single-node run:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/single_node/main_trellis_dualstage_contrastive.sh
```

For a multi-GPU run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash scripts/multi_node/main_trellis_dualstage_contrast.sh
```

The launchers run `edit4shape.systems.trellis.entries.contrastive_dualstage_autograd` with `config/trellis_stage1+2_contrastive.py`. Use `RUN_NAME` and launcher arguments to override the run name or configuration values.

To submit through Koala:

```bash
bash scripts/submit_koala.sh --train
```

For the first run, submit a debug pod, enter it, run the `--download` setup once, and then use `--fast` for later jobs.

## Evaluation

The canonical evaluator compares the pretrained teacher path with the finetuned student path using Dense FlowEdit and CLIP/DINO metrics.

For the documented multi-GPU wrapper:

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

The Python evaluator can also be launched directly with `accelerate` for a single GPU or a custom process layout.

## Checkpoint export

To export a checkpoint with the project helper:

```bash
bash scripts/export/export_checkpoint.sh \
  /path/to/checkpoint \
  /path/to/output
```

The Python implementation is available at [`scripts/export/export_checkpoint.py`](scripts/export/export_checkpoint.py).

## Local environment without Koala

For a local CUDA environment, use Python 3.12 and CUDA 12.4. Install PyTorch first, then the CUDA-dependent packages, then the pinned requirements:

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
pip install -r requirements.txt
```

Set these variables when needed:

```bash
export ATTN_BACKEND=flash_attn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1       # only when all required assets are local
export CUDA_VISIBLE_DEVICES=0
```

The Koala setup script remains the reference implementation for CUDA extensions, cache layout, Qwen snapshot validation, and offline restoration.
