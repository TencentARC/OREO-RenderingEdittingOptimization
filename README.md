# OREO: Fidelity Alignment in 3D Generation via On-the-fly Rendering-Editing Optimization

Official code release for **OREO**, an alignment framework that improves the visual fidelity of 3D generators with 2D diffusion priors.

[Project page](https://theericma.github.io/oreo/)

## Overview

OREO uses a render-edit-optimize loop to align a pretrained 3D generator with the visual prior of a 2D image editor. The current release implements the Dual-Stage Contrastive FlowEdit path for TRELLIS.

The method has two main components:

- **Reinforced Editing** renders views from the current 3D output and edits them with a 2D diffusion prior while preserving the rendered structure.
- **Contrastive Distillation** treats edited and original renderings as positive and negative supervision and transfers the resulting fidelity gap to the 3D generator through a latent contrastive objective.

The released training path jointly optimizes the sparse and dense TRELLIS stages. It includes training launchers, the canonical Dense FlowEdit evaluator, model and dataset download helpers, and checkpoint export utilities.

## Repository structure

```text
config/                         Training configuration
edit4shape/                    OREO training, guidance, rollout, and TRELLIS code
scripts/download/              Public model and dataset download helpers
scripts/eval/                  Dense FlowEdit evaluation
scripts/export/                Checkpoint export utilities
scripts/multi_node/            Multi-GPU training launcher
scripts/single_node/           Single-GPU or local training launcher
SETUP.md                       Environment, model, data, and offline setup
```

## Installation and assets

See [`SETUP.md`](SETUP.md) for the verified Python/CUDA environment, public model downloads, dataset preparation, and offline cache setup.

## Training

After completing the setup, run a local single-GPU or small-scale experiment with:

```bash
CUDA_VISIBLE_DEVICES=0 \
bash scripts/single_node/main_trellis_dualstage_contrastive.sh
```

For a multi-GPU run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash scripts/multi_node/main_trellis_dualstage_contrast.sh
```

Both launchers use [`config/trellis_stage1+2_contrastive.py`](config/trellis_stage1+2_contrastive.py) and run `edit4shape.systems.trellis.entries.contrastive_dualstage_autograd`. Set `RUN_NAME` or pass configuration overrides to customize a run.

## Evaluation

The canonical evaluator compares the pretrained teacher path with the finetuned student path using Dense FlowEdit and CLIP/DINO metrics:

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

## Citation

If you use this code or build on OREO, please cite:

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
