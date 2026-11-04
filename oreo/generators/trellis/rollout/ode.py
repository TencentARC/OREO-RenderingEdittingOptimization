"""
ODE Euler rollout for Trellis sparse / dense stages (inference + ReFL/DRaFT-style training).

Optional x0 / x1 / v regularization; supports three-phase Autograd via RolloutTracker.
"""

from typing import Optional, Any
import numpy as np
import torch
from tqdm import tqdm
from accelerate import Accelerator
import ml_collections

from trellis.modules.sparse import SparseTensor

from .base import (
    predict_sparse_velocity_with_cfg, _predict_sparse_cond_velocity, mix_cfg_sparse,
    prepare_embeddings,
    predict_dense_velocity_with_cfg,
)
from .autograd_tracker import RolloutTracker


# =====================================================================
# Type aliases (avoid circular imports)
# =====================================================================
TrellisState = Any
System = Any


# =====================================================================
# Rollout - core sampling loop (training/assessment common)
# =====================================================================

def rollout_sparse(
    state: TrellisState,
    cfg: ml_collections.ConfigDict,
    system: System,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
    is_training: bool = False,
    tracker: Optional[RolloutTracker] = None,
) -> Optional[RolloutTracker]:
    """Sparse SLAT Stage-2 denoising: x_T -> ... -> x_0 (feats) -> denorm onto state.stage2.z0."""
    pipeline = system.pipeline
    slat_steps, slat_guidance, slat_rescale_t, cfg_min, cfg_max, _ = pipeline.sparse.get_runtime_params()

    # ---- 1. initialization ----
    cond_emb, uncond_emb = prepare_embeddings(state, device)

    assert state.coords is not None, "state.coords Missing"
    assert generator is not None, "generator Must be provided by the caller"

    # ★ Use SparseTensor throughout the process (alignment trellis2 implementation)
    x_t = pipeline.sparse.init_latents(
        coords=state.coords,
        in_channels=pipeline.sparse.resolve_flow_module().in_channels,
        generator=generator
    )  # SparseTensor

    # ---- 2. Scheduler Configuration ----
    scheduler = pipeline.sparse.scheduler()
    scheduler.set_timesteps(slat_steps, device=device, rescale_t=slat_rescale_t)

    # ---- 3. Regularization configuration ----
    reg_type = cfg.rollout.reg.type
    reg_enabled = reg_type != "none" and (is_training or tracker is not None)

    reg_loss_sum = 0.0

    # ---- 4. denoising loop ----
    steps = list(scheduler.timesteps)[:-1]
    steps_iter = tqdm(steps, desc="Rollout", leave=False,
                      disable=not (is_training or tracker is not None) or not Accelerator().is_main_process)

    B = cond_emb.shape[0]  # ()

    for t in steps_iter:
        t_val = float(t.item())
        t_norm = t_val / 1000.0
        t_batch = torch.full((B,), t_val, device=device, dtype=torch.float32)  # (B,)
        use_cfg = cfg_min <= t_val <= cfg_max

        if tracker is not None:
            # ============================================================
            # ★ three stages Autograd Mode: step-by-step reasoning + proxy Insert
            # ============================================================

            # ---- Conditional speed prediction (no_grad, reasoning only)----
            with torch.no_grad():
                cond_pred = _predict_sparse_cond_velocity(
                    pipeline, x_t, t_batch, cond_emb
                )  # SparseTensor

            # ---- Proxy insert(cond_pred level,CFG before)----
            tracker.timesteps.append(t_val)
            tracker.input_trajectory.append(x_t.feats.detach().clone())  # (N, C)
            cond_proxy = cond_pred.feats.detach().clone().requires_grad_(True)  # (N, C)
            tracker.output_trajectory.append(cond_proxy)
            cond_pred = cond_pred.replace(cond_proxy)  # SparseTensor with proxy feats

            # ---- CFG mix(proxy After that,velocity Depend on cond_proxy)----
            if use_cfg and uncond_emb is not None:
                with torch.no_grad():
                    uncond_pred = _predict_sparse_cond_velocity(
                        pipeline, x_t, t_batch, uncond_emb
                    )  # SparseTensor
                velocity = mix_cfg_sparse(
                    cond_pred, uncond_pred, slat_guidance, uncond_mode="detach"
                )  # SparseTensor,feats Depend on cond_proxy
            else:
                velocity = cond_pred  # SparseTensor,feats == cond_proxy

            # ---- Regularization (velocity Pass proxy chain There is a gradient)----
            if reg_enabled:
                with system.strategy.sparse_teacher_context(), torch.no_grad():
                    teacher_vel = predict_sparse_velocity_with_cfg(
                        pipeline, x_t, t_val, cond_emb, uncond_emb,
                        slat_guidance, cfg_min, cfg_max, device,
                    )  # SparseTensor

                if reg_type == "x0":
                    x0_stu = x_t.feats - t_norm * velocity.feats  # (N, C), depends on proxy
                    x0_tea = x_t.feats - t_norm * teacher_vel.feats  # (N, C)
                    reg_loss = _compute_x0_regularization(x0_stu, x0_tea, t_norm)
                elif reg_type == "x1":
                    x0_stu = x_t.feats - t_norm * velocity.feats  # (N, C), depends on proxy
                    x0_tea = x_t.feats - t_norm * teacher_vel.feats  # (N, C)
                    reg_loss = _compute_x1_regularization(x0_stu, x0_tea)
                elif reg_type == "v":
                    reg_loss = _compute_v_regularization(velocity.feats, teacher_vel.feats)
                else:
                    raise ValueError(f"Unknown reg_type: {reg_type}")

                reg_loss_sum = reg_loss_sum + reg_loss

        elif is_training:
            # ============================================================
            # Original training mode: end-to-end computational graph
            # ============================================================
            velocity = predict_sparse_velocity_with_cfg(
                pipeline, x_t, t_val, cond_emb, uncond_emb,
                slat_guidance, cfg_min, cfg_max, device,
            )  # SparseTensor

            if reg_enabled:
                with system.strategy.sparse_teacher_context(), torch.no_grad():
                    teacher_vel = predict_sparse_velocity_with_cfg(
                        pipeline, x_t, t_val, cond_emb, uncond_emb,
                        slat_guidance, cfg_min, cfg_max, device,
                    )  # SparseTensor

                if reg_type == "x0":
                    x0_stu = x_t.feats - t_norm * velocity.feats  # (N, C)
                    x0_tea = x_t.feats - t_norm * teacher_vel.feats  # (N, C)
                    reg_loss = _compute_x0_regularization(x0_stu, x0_tea, t_norm)
                elif reg_type == "x1":
                    x0_stu = x_t.feats - t_norm * velocity.feats  # (N, C)
                    x0_tea = x_t.feats - t_norm * teacher_vel.feats  # (N, C)
                    reg_loss = _compute_x1_regularization(x0_stu, x0_tea)
                elif reg_type == "v":
                    reg_loss = _compute_v_regularization(velocity.feats, teacher_vel.feats)
                else:
                    raise ValueError(f"Unknown reg_type: {reg_type}")

                reg_loss_sum = reg_loss_sum + reg_loss

        else:
            # ============================================================
            # Inference mode:no_grad
            # ============================================================
            with torch.no_grad():
                velocity = predict_sparse_velocity_with_cfg(
                    pipeline, x_t, t_val, cond_emb, uncond_emb,
                    slat_guidance, cfg_min, cfg_max, device,
                )  # SparseTensor

        # ---- Scheduler Stepping (using SparseTensor)----
        x_t = scheduler.step(velocity, t, x_t).prev_sample  # SparseTensor

    # ---- 5. denormalization ----
    norm = pipeline.pipe.slat_normalization
    std = torch.tensor(norm['std'])[None].to(device)   # (1, C)
    mean = torch.tensor(norm['mean'])[None].to(device)  # (1, C)
    denorm_feats = x_t.feats * std + mean  # (N, C)

    # ---- 6. mount to state ----
    state.stage2.z0 = x_t.replace(denorm_feats)  # SparseTensor with denormalized feats

    # ---- 7. Regularization ----
    num_steps = max(1, len(steps))
    if reg_enabled:
        reg_loss_avg = reg_loss_sum / num_steps

        if tracker is not None and len(tracker.output_trajectory) > 0:
            # ★ Three-stage mode: use in advance autograd.grad Calculate it reg Gradient, save tracker
            reg_grads = torch.autograd.grad(
                reg_loss_avg,
                tracker.output_trajectory,
                retain_graph=True,  # slat share proxy chain,Phase 2 Still need
            )
            tracker.reg_grads = [g.detach().clone() for g in reg_grads]  # T × (N, C)
            tracker.reg_loss_val = reg_loss_avg.item()
            # Reserve reg_loss(with pictures) provided Phase 2 merge backward
            state.stage2.reg_loss = reg_loss_avg
        else:
            state.stage2.reg_loss = reg_loss_avg
    else:
        state.stage2.reg_loss = None

    return tracker


# =====================================================================
# Rollout Dense — Stage 1 (sparse_structure_flow_model)
# =====================================================================

def rollout_dense(
    state: TrellisState,
    cfg: ml_collections.ConfigDict,
    system: System,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> None:
    """Dense Stage-1 denoising on sparse_structure_flow_model: x_T -> ... -> x_0."""
    pipeline = system.pipeline
    ss_steps, ss_guidance, ss_rescale_t, ss_cfg_min, ss_cfg_max = pipeline.dense.get_runtime_params()

    # ---- 1. initialization ----
    cond_emb, uncond_emb = prepare_embeddings(state, device)

    assert generator is not None, "generator Must be provided by the caller"
    x_t = pipeline.dense.init_latents(batch_size=1, generator=generator)  # (1, C, R, R, R)

    # ---- 2. time step sequence ----
    _, t_pairs = pipeline.dense.scheduler(ss_steps, ss_rescale_t)

    # ---- 3. denoising loop ----
    with torch.no_grad():
        for t, t_prev in t_pairs:
            t_val = float(t)
            velocity = predict_dense_velocity_with_cfg(
                pipeline, x_t, t_val, cond_emb, uncond_emb,
                ss_guidance, ss_cfg_min, ss_cfg_max, device,
            )  # (B, C, R, R, R)

            # Euler step: x_{t-1} = x_t - (t - t_prev) * v
            delta = t_val - float(t_prev)
            x_t = x_t - delta * velocity

    # ---- 4. mount to state(none normalization)----
    state.stage1.z0 = x_t  # (B, C, R, R, R)


# =====================================================================
# Regularization function (x0 / v)
# =====================================================================

def _compute_x0_regularization(
    x0_student: torch.Tensor,
    x0_teacher: torch.Tensor,
    t_norm: float,
    eps: float = 1e-4,
) -> torch.Tensor:
    """x0 reg: MSE(x0_stu, x0_tea) / t^2 so scale matches v-regularization."""
    # /t^2 keeps x0-reg on a similar scale to v-reg.
    diff = x0_student - x0_teacher.detach()  # (N, C)
    mse = (diff ** 2).mean()  # scalar
    return mse / (t_norm ** 2 + eps)  # scalar


def _compute_x1_regularization(
    x0_student: torch.Tensor,
    x0_teacher: torch.Tensor,
) -> torch.Tensor:
    """x1 reg: MSE(x0_stu, x0_tea) without /t^2."""
    diff = x0_student - x0_teacher.detach()  # (N, C)
    return (diff ** 2).mean()  # scalar


def _compute_v_regularization(
    v_student: torch.Tensor,
    v_teacher: torch.Tensor,
) -> torch.Tensor:
    """v reg: MSE(v_stu, v_tea); grads flow only into the student velocity head."""
    diff = v_student - v_teacher.detach()  # (N, C)
    return (diff ** 2).mean()  # scalar
