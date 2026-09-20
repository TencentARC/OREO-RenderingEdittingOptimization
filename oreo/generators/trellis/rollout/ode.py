"""
ODE Rollout - Standard Euler sampling

for reasoning and ReFL/DRaFT style training (optional x0/v regularization).
Support three stages Autograd mode (via tracker parameters).
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
    """
    Sparse feature denoising sampling (SLAT Stage 2).

    core process: x_T (noise) → iterative denoising → x_0 (Features) → denormalization

    three stages Autograd mode(tracker is not None):
        - model reasoning in torch.no_grad() Execute next (is_training should be set to False)
        - cond_pred Insert on proxy node(requires_grad=True)
        - proxy → CFG → velocity → scheduler Build lightweight proxy chain
        - reg_loss Pass proxy chain Connect to autograd Figure
        - reg gradient pass autograd.grad Calculate in advance and deposit tracker.reg_grads

    Args:
        state: TrellisState Status object, including conditional encoding, coordinates, etc.
        cfg: configuration object,cfg.rollout.reg.type ("none"|"x0"|"v")
        system: system components (pipeline,renderer etc.)
        device: Run the device
        generator: Random number generator (for reproducibility)
        is_training: Whether it is training mode (tracker mode should be False)
        tracker: three stages Autograd of proxy logger. enabled on incoming proxy chain mode.

    Returns:
        tracker: If passed in tracker, returns the filled tracker; Otherwise return None.

    Side Effects:
        - state.stage2.z0: Mount the denormalized SparseTensor
        - state.stage2.reg_loss: mount reg_loss
    """
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
    """
    Dense Feature denoising sampling (Stage 1 — sparse_structure_flow_model).

    core process: x_T (noise) → iterative denoising → x_0 (z_s)
    Stage 1 None normalization,raw latent directly as z₀.

    Only inference mode is supported (no_grad), used for contrastive Obtained during training teacher z_s.

    Args:
        state: Status object, containing conditional encoding
        cfg: Configuration object
        system: system components (pipeline etc.)
        device: Run the device
        generator: random number generator

    Side Effects:
        - state.stage1.z0: mount Dense Tensor (B, C, R, R, R)
    """
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
    """
    x0 regularization Loss:MSE(x0_stu, x0_tea) / t²

    divide by t² make it with v Regularization is numerically equivalent.
    Gradient can flow to history steps (via x_t accumulated calculation graph).

    Args:
        x0_student: Predicted by the student model x0 (N, C)
        x0_teacher: Predicted by the teacher model x0 (N, C)
        t_norm: normalized time step (0~1)
        eps: Prevent division by zero

    Returns:
        loss: scalar
    """
    diff = x0_student - x0_teacher.detach()  # (N, C)
    mse = (diff ** 2).mean()  # scalar
    return mse / (t_norm ** 2 + eps)  # scalar


def _compute_x1_regularization(
    x0_student: torch.Tensor,
    x0_teacher: torch.Tensor,
) -> torch.Tensor:
    """
    x1 regularization Loss:MSE(x0_stu, x0_tea), not divided by t².

    with x0 The difference between regularization: remove 1/t² zoom,
    make small t(Close to x0) will not be amplified.

    Args:
        x0_student: Predicted by the student model x0 (N, C)
        x0_teacher: Predicted by the teacher model x0 (N, C)

    Returns:
        loss: scalar
    """
    diff = x0_student - x0_teacher.detach()  # (N, C)
    return (diff ** 2).mean()  # scalar


def _compute_v_regularization(
    v_student: torch.Tensor,
    v_teacher: torch.Tensor,
) -> torch.Tensor:
    """
    v regularization Loss:MSE(v_stu, v_tea)

    Do it directly to the velocity field MSE, the gradient only flows to the model call of the current step.

    Args:
        v_student: The speed of student model predictions (N, C)
        v_teacher: The speed of teacher model predictions (N, C)

    Returns:
        loss: scalar
    """
    diff = v_student - v_teacher.detach()  # (N, C)
    return (diff ** 2).mean()  # scalar
