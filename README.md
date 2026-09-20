# OREO: Dual-Stage Contrastive FlowEdit

面向 TRELLIS 的 Dual-Stage Contrastive FlowEdit 训练与评测代码。

## Setup

环境、模型、数据准备、训练、评估和 checkpoint 导出说明见 [`SETUP.md`](SETUP.md)。

## Quick start

```bash
bash scripts/submit_koala.sh --train
```

本地单机试跑：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/single_node/main_trellis_dualstage_contrastive.sh
```
