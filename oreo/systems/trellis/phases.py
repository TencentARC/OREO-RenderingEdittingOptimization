"""
Phase-3 VJP backward for Trellis three-phase autograd (O(1) memory per step).

Phases 0/1/2 live in stage_ops.py and autograd_template.py.

VJP: recomputes cond f_theta only, merges guidance_grad + reg_weight*reg_grad, then
(v_grad * cond_pred.feats).sum().backward().
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
    """Recompute cond velocity per step -> VJP into theta.grad; release graph immediately."""
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

            # ---- VJP: (v_grad * cond_pred.feats).sum().backward() ----
            # Graph holds only this f_theta call; freed right after backward.
            (v_grad * cond_pred.feats).sum().backward()  # theta.grad += ...
        except torch.cuda.OutOfMemoryError:
            logging.warning(
                f"P3 VJP step {i}/{T} OOM → partial grad"
            )
            break

    torch.cuda.empty_cache()
