"""
Trellis system factory — build Pipeline,Renderer,Strategy,Guidance,Optimizer.

This module only contains:
1. _CONFIG:           absl config_flags Configuration entry
2. build_system:      System component factory function
3. build_dataloaders: training/Assessment DataLoader factory

forward/rendering/For evaluation logic see forward.py;Three stages Autograd see phases.py;
See you at the training entrance entries/ subdirectory.
"""

# =====================================================================
# Standard library import
# =====================================================================
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

# =====================================================================
# Third-party library import
# =====================================================================
import numpy as np
import ml_collections
from ml_collections import config_flags

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

# =====================================================================
# TRELLIS / TripoSF Reference implementation path settings (must be before related imports)
# =====================================================================
repo_root = os.path.abspath(os.getcwd())
trellis_ref_root = os.path.join(repo_root, "_reference_codes", "TRELLIS")
if trellis_ref_root not in sys.path:
    sys.path.insert(0, trellis_ref_root)
triposf_ref_root = os.path.join(repo_root, "_reference_codes", "TripoSF")
if triposf_ref_root not in sys.path:
    sys.path.insert(0, triposf_ref_root)

# =====================================================================
# Import within project
# =====================================================================
from oreo.datasets.trellis import (
    TrellisDataConfig,
    TrellisDataModule,
    TrellisCameraTrainConfig,
    TrellisCameraEvalConfig,
)
from oreo.systems.base import compute_guidance_device

# Pipeline
from oreo.generators.trellis.pipeline_adapter import build_pipeline_from_reference

# Renderer
from oreo.renderers.base_renderer import RenderConfig
from oreo.renderers.gaussian_splatting_trellis import GaussianRenderer
from oreo.renderers.sparseflex_trellis import TrellisMeshRasterizer

# Strategy & Optimizer
from oreo.generators.trellis.training_adpter import (
    register_sparse_linear_with_peft,
    inject_lora_to_slat,
    build_optimizer_for_sparse,
    build_optimizer_for_dense,
    TrellisFullFinetuneStrategy,
    TrellisLoRAStrategy,
    TrellisFrozenStrategy,
)

# Use absl of config_flags Manage configuration files
_CONFIG = config_flags.DEFINE_config_file("config", help_string="Path to the config file.")


# =====================================================================
# TrellisSystem — Trellis v1 Dedicated system container
# =====================================================================

@dataclass
class TrellisSystem:
    """
    Trellis v1 System core component container.

    with Trellis2System Parallel independent design:
    - pipeline:    Shared build pipeline
    - renderers:   Renderer dictionary, selected by preserving configuration mesh or gs
    - guidance:    Guidance module
    - optimizer_sparse:   Sparse optimizer(slat_flow_model)
    - optimizer_dense: Dense optimizer(sparse_structure_flow_model), only Dual-Stage Use
    - strategy:    training strategy (LoRA / Full / Frozen)
    - cfg:         Runtime configuration (for StageOps Query)
    - accelerator: Accelerate accelerator (for StageOps Query)
    """

    pipeline: Any = None
    renderers: Dict[str, Any] = field(default_factory=dict)  # "mesh"/"gs" → renderer
    guidance: Any = None
    optimizer_sparse: Any = None
    optimizer_dense: Any = None
    strategy: Any = None  # TrainingStrategy Example
    cfg: Any = None
    accelerator: Any = None  # Accelerator

    @staticmethod
    def setup_env_and_seed(cfg: Any) -> None:
        """
        Set random seeds and deterministic running environment.

        To ensure experiment reproducibility, set the following random source seeds:
        - Python random / NumPy / PyTorch CPU+CUDA / cuDNN
        """
        seed = int(cfg.seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def prepare_lora(
        self,
        cfg: Any,
        adapter: str = "base",
        load_path: Optional[str] = None,
        clone_from: Optional[str] = None,
    ) -> "TrellisSystem":
        """Prepare LoRA adapter."""
        target_modules = [m for m in [self.pipeline, self.guidance] if hasattr(m, "set_adapter")]
        for module in target_modules:
            if load_path and hasattr(module, "load_adapter"):
                module.load_adapter(load_path, adapter_name=adapter)
            module.set_adapter(adapter)
        return self

    def prepare_models_and_optimizers(self, cfg: Any, accelerator: Accelerator) -> "TrellisSystem":
        """
        Pass strategy.prepare_sparse() / prepare_dense() Register model to DDP and wrap the optimizer.

        Even if optimizer for None(eval_only mode), also need to call prepare
        Register the model to accelerator, otherwise accelerator.load_state() Unable to restore weight.

        Sparse (slat_flow_model) always first prepare.
        Dense (sparse_structure_flow_model) enable in configuration dense optimizer time later
        prepare;eval_only even if next optimizer_dense for None The model must also be registered, ensuring
        Accelerate of flat checkpoint The order is stable as model → model_1.
        """
        if accelerator is None:
            return self
        if self.strategy is not None:
            self.optimizer_sparse = self.strategy.prepare_sparse(accelerator, self.optimizer_sparse)
            if cfg.train.dense_optimizer:
                self.optimizer_dense = self.strategy.prepare_dense(
                    accelerator, self.optimizer_dense,
                )
        return self


# =====================================================================
# backward compatible re-export
# (External scripts may from oreo.systems.trellis.system import trellis_forward etc.)
# =====================================================================
from oreo.systems.trellis.forward import (  # noqa: F401
    decode_and_render_mesh,
    decode_and_render_gs,
    trellis_forward,
    evaluate,
)


# =====================================================================
# build_system public subfunction
# =====================================================================

def _build_pipeline(cfg: ml_collections.ConfigDict, accelerator: Accelerator):
    """
    build Pipeline(core build pipeline).

    Pipeline Responsible for: conditional coding, structure/Feature sampling and decoding.

    Args:
        cfg: Complete configuration object
        accelerator: Accelerate Distributed training accelerator

    Returns:
        built Pipeline Example
    """
    return build_pipeline_from_reference(cfg, accelerator)


def _build_renderer_of_type(cfg: ml_collections.ConfigDict, device: str, renderer_type: str):
    """
    Build a single of the specified type Renderer.

    near/far from cfg.renderer.{renderer_type}.near/far Read.

    Args:
        cfg: Full configuration object (read renderer.resolution and other public parameters)
        device: Device string, such as "cuda:0"
        renderer_type: "mesh" or "gs"

    Returns:
        Renderer instance
    """
    per_renderer = cfg.renderer[renderer_type]
    render_cfg = RenderConfig(
        resolution=cfg.renderer.resolution,
        near=per_renderer.near,
        far=per_renderer.far,
        ssaa=cfg.renderer.ssaa,
        bg_color=per_renderer.bg_color,
    )

    if renderer_type == "gs":
        return GaussianRenderer(config=render_cfg, device=device)
    else:
        return TrellisMeshRasterizer(config=render_cfg, device=device)


def _build_renderer(cfg: ml_collections.ConfigDict, device: str):
    """
    Backwards Compatibility: Press cfg.renderer.type Build a single Renderer.
    """
    return _build_renderer_of_type(cfg, device, cfg.renderer.type)


def _build_strategy(
    cfg: ml_collections.ConfigDict,
    pipeline,
    accelerator: Accelerator,
):
    """
    build Strategy(Training strategy:LoRA / Full / Frozen).

    Strategy Determined parameter freezing/thaw, student/How the teacher model is placed.
    Both training and inference modes need to be called.

    Args:
        cfg: Complete configuration object
        pipeline: Built Pipeline Example
        accelerator: Accelerate accelerator

    Returns:
        Already setup of Strategy Example
    """
    train_mode = cfg.train.mode
    train_device = accelerator.device
    teacher_device = (
        train_device if cfg.eval_only
        else compute_guidance_device(accelerator.device)
    )

    if train_mode == "full":
        strategy = TrellisFullFinetuneStrategy(
            pipeline, train_device, teacher_device, cfg.pretrained.model
        )
    elif train_mode == "lora":
        register_sparse_linear_with_peft()
        inject_lora_to_slat(pipeline, cfg.lora)
        strategy = TrellisLoRAStrategy(pipeline, train_device, teacher_device)
    else:
        strategy = TrellisFrozenStrategy(pipeline, train_device, teacher_device)

    strategy.setup()
    return strategy


def _build_training_components_sparse(
    cfg: ml_collections.ConfigDict,
    pipeline,
    strategy,
    accelerator: Accelerator,
    guidance_factory: callable,
):
    """
    build Sparse Training components:Guidance,Gradient Checkpointing,Sparse Optimizer.

    if cfg.eval_only for True, return (None, None).

    Returns:
        (guidance, optimizer_sparse) tuple
    """
    if cfg.eval_only:
        return None, None

    # 1. Create using factory function Guidance
    guidance = guidance_factory(cfg.guidance, train_device=accelerator.device)

    # 2. enable slat_flow_model of Gradient Checkpointing
    slat_model = pipeline.sparse.resolve_flow_module()
    for block in slat_model.blocks:
        block.use_checkpoint = True

    # 3. also for slat_decoder_gs enable Gradient Checkpointing
    decoder_gs = pipeline.pipe.models.get('slat_decoder_gs')
    if decoder_gs is not None and hasattr(decoder_gs, 'blocks'):
        for block in decoder_gs.blocks:
            block.use_checkpoint = True

    # 4. for Sparse Student model creation optimizer
    optimizer_sparse = build_optimizer_for_sparse(strategy.student, cfg.train.optimizer)

    return guidance, optimizer_sparse


def _build_training_components_dense(
    cfg: ml_collections.ConfigDict,
    pipeline,
):
    """
    build Dense Training component: unfreezing parameters,Gradient Checkpointing,Dense Optimizer.

    only in cfg.train.dense_optimizer for True Create an optimizer, otherwise return None.

    Returns:
        optimizer_dense or None
    """
    if cfg.eval_only or not cfg.train.dense_optimizer:
        return None

    ss_model = pipeline.dense.resolve_flow_module()
    # Thaw Dense Model parameters
    for p in ss_model.parameters():
        p.requires_grad = True
    # enable Gradient Checkpointing
    if hasattr(ss_model, 'blocks'):
        for block in ss_model.blocks:
            block.use_checkpoint = True

    optimizer_dense = build_optimizer_for_dense(ss_model, cfg.train.optimizer)

    return optimizer_dense


# =====================================================================
# Build function - System component factory
# =====================================================================

def build_system(
    cfg: ml_collections.ConfigDict,
    accelerator: Accelerator,
    guidance_factory: callable,
) -> TrellisSystem:
    """
    Build a complete Trellis System (orchestration entry).

    Call each sub-function in turn to build:
    1. _build_pipeline   → Pipeline
    2. _build_renderer   → Renderer(Put in renderers dict)
    3. _build_strategy   → Strategy
    4. _build_training_components_sparse → Guidance + Sparse Optimizer
    5. _build_training_components_dense  → Dense Optimizer(optional)

    Args:
        cfg: Complete configuration object
        accelerator: Accelerate Distributed training accelerator
        guidance_factory: Guidance Factory function

    Returns:
        TrellisSystem: System instance containing all components
    """
    device = str(accelerator.device)

    pipeline = _build_pipeline(cfg, accelerator)
    renderer = _build_renderer(cfg, device)
    renderer_key = cfg.renderer.type  # "mesh" or "gs"
    strategy = _build_strategy(cfg, pipeline, accelerator)
    guidance, optimizer_sparse = _build_training_components_sparse(
        cfg, pipeline, strategy, accelerator, guidance_factory,
    )
    optimizer_dense = _build_training_components_dense(cfg, pipeline)

    return TrellisSystem(
        pipeline=pipeline,
        renderers={renderer_key: renderer},
        guidance=guidance,
        optimizer_sparse=optimizer_sparse,
        optimizer_dense=optimizer_dense,
        strategy=strategy,
        cfg=cfg,
        accelerator=accelerator,
    )


def build_flowedit_system(
    cfg: ml_collections.ConfigDict,
    accelerator: Accelerator,
    guidance_factory: callable,
) -> TrellisSystem:
    """
    build FlowEdit training system — Pretrained Rollout + Finetuned Single step denoising.

    with build_system Same, but make sure:
    - Use gs renderer(cfg.renderer.type = "gs")
    - decoder Does not participate in optimization but retains the calculation graph

    Args:
        cfg: Complete configuration object
        accelerator: Accelerate Distributed training accelerator
        guidance_factory: Guidance Factory function

    Returns:
        TrellisSystem: System instance
    """
    return build_system(cfg, accelerator, guidance_factory)


def build_dataloaders(cfg: ml_collections.ConfigDict, accelerator: Accelerator) -> Tuple[DataLoader, DataLoader]:
    """
    Construct training and evaluation DataLoader.

    Args:
        cfg: Configuration object
        accelerator: Accelerate accelerator

    Returns:
        tuple: (train_loader, eval_loader)
    """
    train_cam_cfg = TrellisCameraTrainConfig(
        n_view=cfg.data.train.n_view,
        yaw_range=list(cfg.data.train.yaw_range),
        pitch_range=list(cfg.data.train.pitch_range),
        r_range=list(cfg.data.train.r_range),
        fov_range=list(cfg.data.train.fov_range),
        adaptive_distance=cfg.data.train.adaptive_distance,
    )

    eval_cam_cfg = TrellisCameraEvalConfig(
        n_view=cfg.data.eval.n_view,
        yaw_range=list(cfg.data.eval.yaw_range),
        pitch_range=list(cfg.data.eval.pitch_range),
        r_range=list(cfg.data.eval.r_range),
        fov_range=list(cfg.data.eval.fov_range),
        adaptive_distance=cfg.data.eval.adaptive_distance,
    )

    dm_cfg = TrellisDataConfig(
        batch_size=cfg.data.train.batch_size,
        eval_batch_size=cfg.data.eval.batch_size,
        width=cfg.renderer.resolution,
        height=cfg.renderer.resolution,
        image_dataset_dir=cfg.data.train.dir if not cfg.eval_only else cfg.data.eval.dir,
        eval_image_path=cfg.data.eval.dir,
        train=train_cam_cfg,
        eval=eval_cam_cfg,
    )

    dm = TrellisDataModule(
        dm_cfg,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
    )
    dm.setup()

    train_loader = dm.train_dataloader() if not cfg.eval_only else None
    eval_loader = dm.eval_dataloader()
    return train_loader, eval_loader
