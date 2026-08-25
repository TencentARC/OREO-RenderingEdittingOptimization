# OREO: Dual-Stage Contrastive FlowEdit

面向 TRELLIS 的 Dual-Stage Contrastive FlowEdit 训练与评测代码。

## 环境配置

### 1. 一键安装（推荐）
```bash
conda create -n grpo3d python=3.10 -y && conda activate grpo3d
pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# 可选：加速注意力（按需）
# export TORCH_CUDA_ARCH_LIST="80;86;89;90"
# pip install --no-build-isolation flash-attn==2.8.3
```

### 2. 下载预训练模型

#### 2.1 3D Generators

##### Trellis
- 当前实现无需单独下载权重（随代码使用内置/引用模型）。

### 注意力后端兼容性说明
| 组合 | 状态 | 备注 |
|------|------|------|
| torch 2.5.1+cu124 + flash-attn 2.8.3 | ✅ 稳定 | 已验证前向 (fp16) 正常 |
| torch 2.6.0 + flash-attn 2.8.x | ❌ 失败 | C++ undefined symbol (c10::Error) |
| torch 2.6.0 + xformers 0.0.29.post3 | ✅ 可用 | 若需此组合请单独新环境 |
