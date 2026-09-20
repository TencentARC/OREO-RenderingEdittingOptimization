# OREO: Fidelity Alignment in 3D Generation via On-the-fly Rendering-Editing Optimization

**Zhiyuan Ma<sup>1,2</sup> · Wenbo Hu<sup>2</sup> · Wang Zhao<sup>2</sup> · Pengfei Wang<sup>1</sup> · Ying Shan<sup>2</sup> · Lei Zhang<sup>1</sup>**

<sup>1</sup> The Hong Kong Polytechnic University · <sup>2</sup> Tencent ARC Lab

[Project Page](https://theericma.github.io/oreo/)

Official code release for **OREO**, an alignment framework that improves the visual fidelity of 3D generators with 2D diffusion priors.

## 🌟 Overview

Despite recent progress in 3D generation, pretrained 3D generators can still produce assets with limited visual fidelity. OREO addresses this gap with an on-the-fly render-edit-optimize loop:

1. **Reinforced Editing** renders views from the current 3D output and uses a 2D diffusion editor to improve their visual fidelity while preserving geometric structure.
2. **Contrastive Distillation** treats edited and original renderings as positive and negative supervision, then transfers the fidelity gap to the 3D generator with a latent contrastive objective.

This release implements the **Dual-Stage Contrastive FlowEdit** path for TRELLIS, including sparse and dense stage training, Dense FlowEdit evaluation, public model and dataset download helpers, and checkpoint export utilities.

## 📁 Repository structure

```text
config/                         Training configurations
edit4shape/                    OREO guidance, rollout, renderer, and TRELLIS systems
scripts/download/              Public model and dataset download helpers
scripts/eval/                  Dense FlowEdit evaluation
scripts/export/                Checkpoint export utilities
scripts/multi_node/            Multi-GPU training launcher
scripts/single_node/           Single-GPU or local training launcher
SETUP.md                       Environment, model, data, and offline setup
```

## ⚙️ Installation and assets

Please follow [`SETUP.md`](SETUP.md) for the verified Python/CUDA environment, public model downloads, dataset preparation, and offline cache setup.

## 🚀 Quick start

After completing the setup, run a local single-GPU experiment with:

```bash
CUDA_VISIBLE_DEVICES=0 \
bash scripts/single_node/main_trellis_dualstage_contrastive.sh
```

For a multi-GPU run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash scripts/multi_node/main_trellis_dualstage_contrast.sh
```

Both launchers use [`config/trellis_stage1+2_contrastive.py`](config/trellis_stage1+2_contrastive.py) and run `edit4shape.systems.trellis.entries.contrastive_dualstage_autograd`.

## 🚋 Training

The retained training path jointly optimizes the sparse and dense TRELLIS stages. To customize an experiment, set `RUN_NAME` or pass configuration overrides to the launcher:

```bash
RUN_NAME=my_experiment \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/multi_node/main_trellis_dualstage_contrast.sh
```

## 📊 Evaluation

The canonical evaluator compares the pretrained teacher path with the finetuned student path using Dense FlowEdit and CLIP/DINO metrics:

```bash
bash scripts/eval/eval_trellis_dense_flowedit_DDP.sh \
  /path/to/checkpoints/checkpoint_10_3157 alphaimages
```

The evaluation wrapper also accepts `alphaimage_train`, `gso_selected`, and `toys4k`. FlowEdit parameters can be overridden after the dataset name:

```bash
bash scripts/eval/eval_trellis_dense_flowedit_DDP.sh \
  /path/to/checkpoints/checkpoint_10_3157 gso_selected \
  --config.rollout.flowedit.n_max=12 \
  --config.rollout.flowedit.cfg_scale_tgt=4.0 \
  --config.rollout.flowedit.cfg_scale_src=-4.0
```

## 📦 Checkpoint export

```bash
bash scripts/export/export_checkpoint.sh \
  /path/to/checkpoint \
  /path/to/output
```

## 📚 Citation

If you use OREO or this codebase in your research, please cite:

```bibtex
@inproceedings{ma2026oreo,
  title     = {OREO: Fidelity Alignment in 3D Generation via On-the-fly Rendering-Editing Optimization},
  author    = {Ma, Zhiyuan and Hu, Wenbo and Zhao, Wang and Wang, Pengfei and Shan, Ying and Zhang, Lei},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

## License

This project is released under the [MIT License](LICENSE).
