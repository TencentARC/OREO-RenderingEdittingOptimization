"""
Trellis Dual Contrastive Training entrance — Sparse + Dense dual stage latent Spatial contrastive learning.

Training process (two-stage serial, reducing peak video memory):
  ── Sparse Stage → optimizer_sparse.step ──
  Phase 0–5d: execute shared sparse contrastive phase, output c_src / c_tgt
  → clip_grad_norm + optimizer_sparse.step + zero_grad(release sparse gradient)

  ── Dense Stage → optimizer_dense.step ──
  D1–D3c: Reuse Sparse Output state.stage1.z0 + c_src / c_tgt
  → clip_grad_norm + optimizer_dense.step + zero_grad(release dense gradient)

Peak memory = max(sparse_grad, dense_grad), rather than the sum of the two.

Gradient propagation:
  Sparse: contrastive_loss → v_proxy → v_student → slat_flow_model.grad
  Dense:  contrastive_loss → v_proxy → v_student → sparse_structure_flow_model.grad
"""

# =====================================================================
# standard library + Third-party libraries
# =====================================================================
import gc
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import ml_collections
from absl import app
from accelerate import Accelerator

# =====================================================================
# TRELLIS Reference implementation path
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
from oreo.systems.base import (
    TrainModeGuard,
    CheckpointIO,
    build_run_paths,
)
from oreo.systems.utils import MetricLogger
from oreo.systems.utils.visual import TrellisVisualIO
from oreo.systems.utils.profiler import PhaseProfiler
from oreo.generators.trellis.state import TrellisContrastiveState
from oreo.guidance import create_guidance

from oreo.systems.trellis.system import (
    TrellisSystem,
    _CONFIG,
    build_flowedit_system,
    build_dataloaders,
)
from oreo.systems.trellis.forward import evaluate
from oreo.systems.trellis.autograd_template import (
    trellis_sparse_contrastive_step,
    trellis_dense_contrastive_step,
)
from oreo.systems.trellis.stage_ops import TrellisContrastiveOps, TrellisDenseContrastiveOps

# =====================================================================
# Main function entry
# =====================================================================

def main(argv) -> None:
    """
    Dual Contrastive Main entrance to training.
    """
    del argv
    cfg = _CONFIG.value

    # =====================================================
    # Step 1: Environment settings
    # =====================================================
    TrellisSystem.setup_env_and_seed(cfg)

    # =====================================================
    # Step 2: initialization Accelerator
    # =====================================================
    accelerator = Accelerator(
        mixed_precision=cfg.mixed_precision,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        log_with=["wandb"] if cfg.use_wandb else None,
    )

    # =====================================================
    # Step 3: Create run directory
    # =====================================================
    run_root, logs_dir, visuals_train_dir, visuals_eval_dir = build_run_paths(cfg, accelerator)

    if cfg.use_wandb and accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="trellis_oreo",
            config=dict(cfg),
            init_kwargs={"wandb": {"name": cfg.run_name}},
        )

    vis_freq = int(cfg.freq.save.visual)
    visual_io = TrellisVisualIO(
        visuals_train_dir,
        target_h=cfg.renderer.resolution,
        vis_freq=vis_freq,
        accelerator=accelerator,
    )

    # =====================================================
    # Step 4: Build data loader
    # =====================================================
    train_loader, eval_loader = build_dataloaders(cfg, accelerator)

    # =====================================================
    # Step 5: Build system components
    # =====================================================
    system = build_flowedit_system(cfg, accelerator, guidance_factory=create_guidance)
    system = system.prepare_lora(cfg, adapter="base", load_path=None, clone_from=None)
    system = system.prepare_models_and_optimizers(cfg, accelerator)

    # =====================================================
    # Step 6: Checkpoint management
    # =====================================================
    ckpt_root = run_root / "checkpoints"
    ckpt_io = CheckpointIO(accelerator, ckpt_root)
    start_epoch = ckpt_io.load(cfg.checkpoint, mode="train")
    global_step = int(ckpt_io.start_global_step)

    # =====================================================
    # Step 7: Assessment mode
    # =====================================================
    if cfg.eval_only:
        eval_log = evaluate(
            system, cfg, accelerator,
            epoch=start_epoch,
            global_step=global_step,
            eval_loader=eval_loader,
            visuals_eval_dir=visuals_eval_dir,
        )
        eval_logger = MetricLogger(accelerator, logs_dir / "test.csv")
        eval_logger.accumulate(eval_log, 1)
        eval_logger.flush(global_step, start_epoch)
        return

    # =====================================================
    # Step 8: training loop (★ Dual Contrastive)
    # =====================================================
    train_logger = MetricLogger(accelerator, logs_dir / "train.csv")

    pipeline = system.pipeline
    pipe_models = pipeline.pipe.models
    slat_model = pipe_models['slat_flow_model']
    ss_model = pipe_models['sparse_structure_flow_model']

    profiler = PhaseProfiler(
        enabled=True,
        verbose=accelerator.is_main_process,
    )

    sparse_ops = TrellisContrastiveOps()
    dense_ops = TrellisDenseContrastiveOps()

    for epoch in range(start_epoch, int(cfg.num_epochs)):
        train_loader.sampler.set_epoch(epoch)

        for batch in train_loader:
            global_step += 1

            # ── Sparse: forward + backward + optimizer step ──
            with TrainModeGuard(slat_model), TrainModeGuard(ss_model):
                state = TrellisContrastiveState()
                state.attach_batch(batch, pipeline=pipeline)

                sparse_log = trellis_sparse_contrastive_step(
                    sparse_ops, state, system, global_step, profiler,
                )

            accelerator.clip_grad_norm_(slat_model.parameters(), 10.0)
            system.optimizer_sparse.step()
            system.optimizer_sparse.zero_grad()

            # ── Dense: forward + backward + optimizer step ──
            with TrainModeGuard(ss_model):
                dense_log = trellis_dense_contrastive_step(
                    dense_ops, state, system, global_step, profiler,
                )

            accelerator.clip_grad_norm_(ss_model.parameters(), 10.0)
            system.optimizer_dense.step()
            system.optimizer_dense.zero_grad()

            train_log = {**sparse_log, **dense_log}

            # Visual save
            if accelerator.is_main_process and (global_step % visual_io.vis_freq == 0):
                _pipe = system.guidance.pipe if system.guidance else None
                _n_prog = cfg.freq.save.progress_samples
                visual_io.save_contrastive_train(
                    state, epoch, global_step,
                    pipe=_pipe, n_progress_samples=_n_prog,
                )

            # profiler collect + Log merge
            time_log = profiler.collect(global_step, print_freq=10)
            train_log.update(time_log)

            train_logger.log_step(train_log, len(batch['image_pils']), global_step, epoch)

        # ---- periodic evaluation ----
        if cfg.freq.eval and (epoch % int(cfg.freq.eval) == 0):
            eval_log = evaluate(
                system, cfg, accelerator,
                epoch=epoch,
                global_step=global_step,
                eval_loader=eval_loader,
                visuals_eval_dir=visuals_eval_dir,
            )
            eval_logger = MetricLogger(accelerator, logs_dir / "test.csv")
            eval_logger.accumulate(eval_log, 1)
            eval_logger.flush(global_step, epoch)

        # ---- Periodically save checkpoints ----
        if cfg.freq.save.ckpt and (epoch % int(cfg.freq.save.ckpt) == 0):
            ckpt_io.save(system, state, cfg, epoch, global_step)


# =====================================================================
# program entry point
# =====================================================================
if __name__ == "__main__":
    app.run(main)
