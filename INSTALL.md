## 安装与预训练权重准备

## 安装速览（最短路径）

目标环境（已验证）：Python 3.10，torch 2.5.1+cu124，flash‑attn 2.8.3（sm90），kaolin 0.18.0，spconv‑cu124 2.3.8，cumm‑cu124 0.7.11，nvdiffrast 0.3.3。

### 一次安装（复制即可）
```bash
# 创建并激活环境
conda create -n grpo3d python=3.10 -y && \
source /home/zhiyuan_ma/miniconda3/etc/profile.d/conda.sh && conda activate grpo3d

# PyTorch（cu124）
pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 \
  --index-url https://download.pytorch.org/whl/cu124

# 3D 依赖
pip install kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html
pip install spconv-cu124==2.3.8 cumm-cu124==0.7.11

# flash-attn（Hopper 用 90）
export TORCH_CUDA_ARCH_LIST=90 && \
pip install --no-build-isolation flash-attn==2.8.3

# 其余依赖（若 torchsparse 装不上可忽略；nvdiffrast 无轮子则用源码）
pip install -r requirements.txt || true
pip install git+https://github.com/NVlabs/nvdiffrast.git@v0.3.3
```


### 下载 BiRefNet 权重
```bash
export HF_HUB_OFFLINE=0
# 下载 BiRefNet 到 HF 缓存
python ./scripts/download/download_birefnet.py
export HF_HUB_OFFLINE=1
```

### 下载 DINOv2 Giant 权重
```bash
python ./scripts/download/download_dinov2.py \
  --out ./pretrained_weights/dinov2-giant
```

### 可选：安装 torchsparse（源码编译，无需 root）
```bash
# 1) 安装 sparsehash 头文件（conda，无需 root）
conda install -y -c bioconda google-sparsehash

# 2) 确保编译器能找到头文件（当前会话）
export CPLUS_INCLUDE_PATH="$CONDA_PREFIX/include:$CPLUS_INCLUDE_PATH"
export CPATH="$CONDA_PREFIX/include:$CPATH"

# 3) 可选：安装 ninja 提升 C++/CUDA 编译速度
pip install ninja

# 4) 源码安装 torchsparse（与 torch 2.5.1+cu124 组合已实测）
pip install -v git+https://github.com/mit-han-lab/torchsparse.git

# 5) 验证
python -c "import torchsparse; print('OK')"
```

- 说明与注意事项：
  - 若构建时出现 CUDA 12.6 vs 12.4 的小版本提示，可忽略（来自系统 CUDA 工具链 vs PyTorch CUDA 的小版本差异）。
  - 未设置 `TORCH_CUDA_ARCH_LIST` 时，PyTorch 会自动选择可见 GPU 架构；Hopper 可手动：`export TORCH_CUDA_ARCH_LIST=90`。
  - 若仍提示找不到 `google/dense_hash_map`，请确认 `$CONDA_PREFIX/include/google/dense_hash_map` 存在，或重新设置 `CPLUS_INCLUDE_PATH`/`CPATH` 后再编译。
  - 运行时切换后端：`SPARSE_BACKEND=torchsparse`（脚本已支持通过环境变量覆盖）。
  - 未安装 xformers 时：默认使用 flash‑attn；设置 `export ATTN_BACKEND=flash_attn` 即可，无需安装 xformers。

<!-- 
- 无法联网或不登录 W&B 时，可使用离线模式：
  ```bash
  export WANDB_MODE=offline
  export WANDB_DISABLED=1
  ```
 -->
