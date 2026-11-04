"""
Dual-Stage Contrastive FlowEdit autograd step templates (sparse + dense).

Reuses velocity regularization and StageOps; sparse produces c_src/c_tgt reused by dense.
"""

from __future__ import annotations

import contextlib
from typing import Any, Dict

import torch
from oreo.generators.trellis.state import TrellisState
from oreo.generators.trellis.rollout import VelocityTracker
from oreo.systems.trellis.system import TrellisSystem
from oreo.systems.utils.profiler import PhaseProfiler
from oreo.generators.trellis.rollout.ode import (
    _compute_x0_regularization,
    _compute_x1_regularization,
    _compute_v_regularization,
)


# =====================================================================
# CFG override context manager
# =====================================================================

@contextlib.contextmanager
def _disable_uncond(state, denoise_cfg: bool):
    """Temporarily clear uncond_embed to skip uncond forward."""
    if denoise_cfg:
        yield
        return
    original = state.views_conditioned.uncond_embed
    state.views_conditioned.uncond_embed = None
    try:
        yield
    finally:
        state.views_conditioned.uncond_embed = original


# =====================================================================
# Phase 3.5: Shared velocity regularization backward
# =====================================================================

def _phase3_5_velocity_reg(
    ops,
    state: TrellisState,
    system: TrellisSystem,
    tracker: "VelocityTracker",
    zt_feats: torch.Tensor,
    t_val: float,
    reg_weight: float,
    reg_type: str = "v",
) -> None:
    """Phase 3.5: teacher velocity + reg backward -> tracker.reg_grad."""
    v_teacher = ops.predict_velocity_teacher(
        state, system, zt_feats, t_val,
    )  # (N, C), detached

    if reg_type == "x0":
        x0_stu = zt_feats - t_val * tracker.v_proxy  # (N, C), Depend on v_proxy
        x0_tea = zt_feats - t_val * v_teacher         # (N, C), detached
        raw_reg = _compute_x0_regularization(x0_stu, x0_tea, t_val)  # scalar
    elif reg_type == "x1":
        x0_stu = zt_feats - t_val * tracker.v_proxy  # (N, C), Depend on v_proxy
        x0_tea = zt_feats - t_val * v_teacher         # (N, C), detached
        raw_reg = _compute_x1_regularization(x0_stu, x0_tea)  # scalar
    elif reg_type == "v":
        raw_reg = _compute_v_regularization(tracker.v_proxy, v_teacher)  # scalar
    else:
        raise ValueError(
            f"Unknown reg_type: {reg_type!r}, expected 'v', 'x0', or 'x1'"
        )

    reg_loss = reg_weight * raw_reg  # scalar
    reg_loss.backward()  # → v_proxy.grad = reg_grad
    tracker.reg_grad = tracker.v_proxy.grad.detach().clone()  # (N, C)
    tracker.reg_loss_val = reg_loss.item()
    tracker.v_proxy.grad = None  # ★ Clear, give P4c of guidance gradient vacancy

    del v_teacher, reg_loss
    torch.cuda.empty_cache()


# =====================================================================
# Sparse Contrastive FlowEdit Training steps:Latent spatial contrastive learning
# =====================================================================

def trellis_sparse_contrastive_step(
    ops,
    state,
    system: TrellisSystem,
    global_step: int,
    profiler: PhaseProfiler,
    prefix: str = "",
) -> Dict[str, Any]:
    """Sparse contrastive FlowEdit step in latent space (rollout -> render -> FlowEdit -> contrastive)."""
    seed = int(system.cfg.seed)
    cfg = system.cfg
    device = system.accelerator.device

    # ── P0: Dense Sampling ──
    profiler.tick(f"{prefix}P0_pre_rollout")
    ops.pre_rollout(state, system, global_step)

    # ── P1: Rollout → clean z₀ ──
    profiler.tick(f"{prefix}P1_rollout")
    ops.rollout(state, system, seed)
    state.stage2.z0_teacher = state.stage2.z0  # save teacher slat Quote

    # ── P2: Noise z₀ → zₜ ──
    profiler.tick(f"{prefix}P2_add_noise")
    z0_norm = ops.normalize_latent(state, system)  # (N, C)
    t_val = ops.sample_timestep(system)
    zt_feats = ops.add_noise(z0_norm.detach(), t_val)  # (N, C), detached

    # ── P3: Student velocity + proxy ──
    profiler.tick(f"{prefix}P3_velocity")
    student_denoise_cfg = cfg.train.student_denoise_cfg  # True = use CFG, False = skip uncond forward
    with _disable_uncond(state, student_denoise_cfg):
        v_student = ops.predict_velocity_student(state, system, zt_feats, t_val)

    tracker = VelocityTracker()
    tracker.setup_proxy(v_student)

    # ẑ₀_stu = zₜ - t·v_proxy(Gradient via v_proxy leaf)
    z0_hat_norm = zt_feats - t_val * tracker.v_proxy  # (N, C)
    # ★ z0_hat_norm Retained as a local variable and not updated slat(No need decode student)

    state.stage2.z0._spatial_cache.clear()
    torch.cuda.empty_cache()

    # ── P3.5: Velocity reg(optional)──
    reg_weight = float(cfg.train.loss.reg)
    reg_type = str(cfg.train.loss.reg_type)
    if reg_weight > 0:
        profiler.tick(f"{prefix}P3.5_reg")
        with _disable_uncond(state, student_denoise_cfg):
            _phase3_5_velocity_reg(
                ops, state, system, tracker,
                zt_feats, t_val,
                reg_weight=reg_weight,
                reg_type=reg_type,
            )

    # ── P4a: Render Teacher z₀ → src ──
    profiler.tick(f"{prefix}P4a_render_teacher")
    ops.decode_render_teacher(state, system)

    # ── P4b: FlowEdit edit src → tgt ──
    profiler.tick(f"{prefix}P4b_edit_views")
    ops.edit_views(state, system)

    # ── P4c: DINOv2 encode → c_src, c_tgt (mount to state) ──
    profiler.tick(f"{prefix}P4c_encode")
    ops.encode_conditions(state, system)

    # ── P4c.5: DINO similarity Monitor + Optional adaptive swap ──
    dino_sim, sim_src, sim_tgt = ops.compute_dino_similarity(state)
    contrastive_cfg = cfg.train.loss.contrastive
    if bool(contrastive_cfg.adaptive_swap):
        swap_rate = ops.adaptive_swap_conditions(state, sim_src, sim_tgt)
        dino_sim["sim/swap_rate"] = swap_rate

    # ── P5a: Re-noise ẑ₀_stu → zₜ_stu; Teacher denoise with c_tgt → positive ──
    profiler.tick(f"{prefix}P5a_teacher_tgt")
    zt_stu = ops.add_noise(z0_hat_norm.detach(), t_val)  # (N, C), from student Predictive re-noising
    z0_tea_tgt = ops.predict_x0_teacher_with_cond(
        state, system, zt_stu, t_val, cond_source="edited",
    )  # (N, C), detached

    # ── P5b: Teacher denoise zₜ_stu with c_src → negative ──
    profiler.tick(f"{prefix}P5b_teacher_src")
    z0_tea_src = ops.predict_x0_teacher_with_cond(
        state, system, zt_stu, t_val, cond_source="generated",
    )  # (N, C), detached
    del zt_stu

    torch.cuda.empty_cache()

    # ── P5c: Contrastive loss ──
    profiler.tick(f"{prefix}P5c_contrastive_loss")
    guidance_weight = float(cfg.train.loss.guidance)
    ada = bool(contrastive_cfg.ada)
    eps = float(contrastive_cfg.eps)

    loss = ops.contrastive_loss(
        z0_hat_norm, z0_tea_tgt, z0_tea_src, ada=ada, eps=eps,
    )  # scalar
    loss_weighted = loss * guidance_weight
    loss_weighted.backward()  # → v_proxy.grad

    del z0_hat_norm, z0_tea_tgt, z0_tea_src

    # ── P5d: Relay → θ.grad ──
    profiler.tick(f"{prefix}P5d_relay")
    tracker.relay_and_backward()

    profiler.tick(f"{prefix}end")

    # ── Build log ──
    log: Dict[str, Any] = {}
    log[f"{prefix}loss/contrastive"] = loss.item()
    log[f"{prefix}loss/contrastive_weighted"] = loss_weighted.item()
    log.update({
        f"{prefix}{k}": v
        for k, v in tracker.collect_log(reg_weight=reg_weight).items()
    })
    log[f"{prefix}noise/t"] = t_val
    log.update({f"{prefix}{k}": v for k, v in dino_sim.items()})
    log[f"{prefix}dense/teacher_fallback"] = float(state.dense_fallback)

    del tracker, loss, loss_weighted
    torch.cuda.empty_cache()
    return log


# =====================================================================
# Dense Contrastive Step — supply Dual-Stage Entrance independent call
# =====================================================================

# Dense reuses c_src/c_tgt produced by the sparse contrastive step above.

def trellis_dense_contrastive_step(
    dense_ops,
    state,
    system: TrellisSystem,
    global_step: int,
    profiler: PhaseProfiler,
    prefix: str = "",
) -> Dict[str, Any]:
    """Dense contrastive step (D1-D3c) reusing Sparse-stage c_src/c_tgt."""
    log: Dict[str, Any] = {}

    cfg = system.cfg
    dp = f"{prefix}dense/"

    state.stage1.z0_teacher = state.stage1.z0

    # ── D1: Noise z_s → zₜ ──
    profiler.tick(f"{dp}D1_add_noise")
    z0_dense = dense_ops.normalize_latent(state, system)  # (B, C, R, R, R)
    t_val = dense_ops.sample_timestep(system)
    zt_dense = dense_ops.add_noise(z0_dense.detach(), t_val)  # (B, C, R, R, R)

    # ── D2: Student velocity + proxy ──
    profiler.tick(f"{dp}D2_velocity")
    student_denoise_cfg = cfg.train.student_denoise_cfg  # True = use CFG, False = skip uncond forward
    with _disable_uncond(state, student_denoise_cfg):
        v_student_dense = dense_ops.predict_velocity_student(state, system, zt_dense, t_val)

    tracker_dense = VelocityTracker()
    tracker_dense.setup_proxy(v_student_dense)

    z0_hat_dense = zt_dense - t_val * tracker_dense.v_proxy  # (B, C, R, R, R)

    torch.cuda.empty_cache()

    # ── D2.5: Velocity reg(optional)──
    reg_weight = float(cfg.train.loss.reg)
    reg_type = str(cfg.train.loss.reg_type)
    if reg_weight > 0:
        profiler.tick(f"{dp}D2.5_reg")
        with _disable_uncond(state, student_denoise_cfg):
            _phase3_5_velocity_reg(
                dense_ops, state, system, tracker_dense,
                zt_dense, t_val,
                reg_weight=reg_weight,
                reg_type=reg_type,
            )

    # ── D3a: Teacher denoise with c_tgt → positive ──
    profiler.tick(f"{dp}D3a_teacher_tgt")
    zt_stu_dense = dense_ops.add_noise(z0_hat_dense.detach(), t_val)
    z0_tea_tgt = dense_ops.predict_x0_teacher_with_cond(
        state, system, zt_stu_dense, t_val, cond_source="edited",
    )

    # ── D3b: Teacher denoise with c_src → negative ──
    profiler.tick(f"{dp}D3b_teacher_src")
    z0_tea_src = dense_ops.predict_x0_teacher_with_cond(
        state, system, zt_stu_dense, t_val, cond_source="generated",
    )
    del zt_stu_dense

    torch.cuda.empty_cache()

    # ── D3c: Contrastive loss + relay ──
    profiler.tick(f"{dp}D3c_contrastive_loss")
    contrastive_cfg = cfg.train.loss.contrastive
    guidance_weight = float(cfg.train.loss.guidance)
    ada = bool(contrastive_cfg.ada)
    eps = float(contrastive_cfg.eps)

    loss_dense = dense_ops.contrastive_loss(
        z0_hat_dense, z0_tea_tgt, z0_tea_src, ada=ada, eps=eps,
    )
    loss_dense_weighted = loss_dense * guidance_weight
    loss_dense_weighted.backward()

    del z0_hat_dense, z0_tea_tgt, z0_tea_src

    profiler.tick(f"{dp}D3c_relay")
    tracker_dense.relay_and_backward()

    profiler.tick(f"{dp}end")

    # ── Dense Log ──
    log[f"{dp}loss/contrastive"] = loss_dense.item()
    log[f"{dp}loss/contrastive_weighted"] = loss_dense_weighted.item()
    log.update({
        f"{dp}{k}": v
        for k, v in tracker_dense.collect_log(reg_weight=reg_weight).items()
    })
    log[f"{dp}noise/t"] = t_val

    del tracker_dense, loss_dense, loss_dense_weighted
    torch.cuda.empty_cache()
    return log
