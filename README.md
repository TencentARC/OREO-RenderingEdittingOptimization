# OREO: Fidelity Alignment in 3D Generation via On-the-fly Rendering-Editing Optimization

**Zhiyuan Ma<sup>1,2</sup> · Wenbo Hu<sup>2</sup> · Wang Zhao<sup>2</sup> · Pengfei Wang<sup>1</sup> · Ying Shan<sup>2</sup> · Lei Zhang<sup>1</sup>**

<sup>1</sup> The Hong Kong Polytechnic University · <sup>2</sup> Tencent ARC Lab

[Project Page](https://theericma.github.io/oreo/)

Official code for **OREO**, an alignment framework that improves the visual fidelity of 3D generators with 2D diffusion priors.

## Overview

OREO uses an on-the-fly render-edit-optimize loop:

- **Reinforced Editing** improves rendered views with a 2D diffusion editor while preserving structure.
- **Contrastive Distillation** transfers the fidelity gap between edited and original views to the 3D generator.

This release implements Dual-Stage Contrastive FlowEdit for TRELLIS.

## Setup

See [`SETUP.md`](SETUP.md) for the Python/CUDA environment, public model downloads, datasets, and offline setup.

## Training

```bash
CUDA_VISIBLE_DEVICES=0 \
bash scripts/single_node/main_trellis_oreo.sh
```

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash scripts/multi_node/main_trellis_oreo.sh
```

## Evaluation

```bash
bash scripts/eval/eval_trellis_dense_flowedit_DDP.sh \
  /path/to/checkpoints/checkpoint_10_3157 alphaimages
```

## Citation

```bibtex
@inproceedings{ma2026oreo,
  title     = {OREO: Fidelity Alignment in 3D Generation via On-the-fly Rendering-Editing Optimization},
  author    = {Ma, Zhiyuan and Hu, Wenbo and Zhao, Wang and Wang, Pengfei and Shan, Ying and Zhang, Lei},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

## License

[MIT License](LICENSE)
