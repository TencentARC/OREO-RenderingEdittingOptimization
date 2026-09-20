# OREO: Dual-Stage Contrastive FlowEdit

面向 TRELLIS 的 Dual-Stage Contrastive FlowEdit 训练与评测代码。

## Setup

环境、模型、数据准备、训练、评估和 checkpoint 导出说明见 [`SETUP.md`](SETUP.md)。

## Quick start

```bash
CUDA_VISIBLE_DEVICES=0 \
bash scripts/single_node/main_trellis_dualstage_contrastive.sh
```

多 GPU 训练：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash scripts/multi_node/main_trellis_dualstage_contrast.sh
```
