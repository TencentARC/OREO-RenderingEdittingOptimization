"""
Trellis three stages Autograd — Phase 3 VJP Realize.

Phase 3: phase3_rollout_grad_backward
  Gradual recalculation cond f_θ → VJP backward → θ.grad Accumulation (Video Memory O(1))

Others Phase (0/1/2) Migrated to:
  - stage_ops.py        — pre_rollout (inline dense sampling), rollout, decode_render
  - autograd_template.py — Arrange + guidance backward + decode-grad

Design reference:trellis2_distill branched trellis2_shape_autograd_async.py
"""

from __future__ import annotations

import logging

import torch

from oreo.generators.trellis.state import TrellisState
from oreo.generators.trellis.rollout import (
    RolloutTracker,
    _predict_sparse_cond_velocity,
)
from oreo.systems.trellis.system import TrellisSystem

# SparseTensor: TRELLIS used to represent sparseness in 3D Feature core data structure
from trellis.modules.sparse import SparseTensor


# =====================================================================
# Phase 3: VJP Backward
# =====================================================================

def phase3_rollout_grad_backward(
    state: TrellisState,
    system: TrellisSystem,
    cfg,
    device: torch.device,
    tracker: RolloutTracker,
) -> None:
    """
    Phase 3: Gradual recalculation cond f_θ → VJP backward → θ.grad Accumulate.

    Video memory O(1): Each step only retains a single f_θ The calculation graph ofVJP released immediately.

    - v_grad from tracker.output_trajectory[i].grad(Phase 2 backward padding, only guidance gradient)
    - reg_grads from tracker.reg_grads(Phase 1 autograd.grad precomputed)
    - Phase 3 merge both: v_grad = guidance_grad + reg_weight * reg_grad
    - Recalculate conditional forecasts only f_θ(x_t, t), no need uncond / CFG / reg
    - VJP: (v_grad * cond_pred.feats).sum().backward()

    Args:
        state: TrellisState
        system: system components
        cfg: Configuration object
        device: Run the device
        tracker: Phase 1 filled RolloutTracker
    """
    pipeline = system.pipeline
    cond_emb, _ = state.extract_embeddings()
    cond_emb = cond_emb.to(device)  # (B, S, C)
    B = cond_emb.shape[0]  # ()

    T = len(tracker.timesteps)
    has_reg_grads = len(tracker.reg_grads) == T
    reg_weight = cfg.train.loss.reg if has_reg_grads else 0.0

    for i in range(T):
        t_val = tracker.timesteps[i]
        x_t_feats = tracker.input_trajectory[i]  # (N, C), pure data

        # use state.stage2.z0 of coords rebuild SparseTensor(x_t_feats no gradient)
        x_t = SparseTensor(
            feats=x_t_feats,
            coords=state.stage2.z0.coords,
        )  # SparseTensor(feats: (N, C), coords: (N, 4))

        t_batch = torch.full((B,), t_val, device=device, dtype=torch.float32)  # (B,)

        try:
            # ★ Recalculate conditional forecast (with θ gradient,x_t in _predict_sparse_cond_velocity internal detach)
            cond_pred = _predict_sparse_cond_velocity(
                pipeline, x_t, t_batch, cond_emb
            )  # SparseTensor(feats: (N, C))

            # ---- merge guidance + reg gradient ----
            v_grad = tracker.output_trajectory[i].grad  # (N, C) or None
            if v_grad is None:
                continue

            if has_reg_grads:
                v_grad = v_grad + reg_weight * tracker.reg_grads[i]  # (N, C)

            # ---- VJP:(v_grad * cond_pred.feats).sum().backward() ----
            # The picture only contains this time f_θ call,backward released immediately after
            (v_grad * cond_pred.feats).sum().backward()  # θ.grad += ...
        except torch.cuda.OutOfMemoryError:
            logging.warning(
                f"P3 VJP step {i}/{T} OOM → partial grad"
            )
            break

    torch.cuda.empty_cache()
