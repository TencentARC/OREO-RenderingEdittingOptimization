"""Shared rollout helpers: CFG mixing, device transfer, and velocity prediction (dense / sparse)."""

from typing import Any, Optional, Tuple
import torch
from trellis.modules.sparse import SparseTensor


# =====================================================================
# Rollout Helper functions (module level, avoid nesting)
# =====================================================================

def mix_cfg_sparse(
    cond_pred: Any,  # SparseTensor
    uncond_pred: Any,  # SparseTensor
    scale: float,
    uncond_mode: str = "detach",
) -> Any:  # SparseTensor
    """
    Classifier-Free Guidance mix on SparseTensor.
    
    CFG: output = cond_pred + scale * (cond_pred - uncond_pred) (feats-wise).
    """
    if uncond_pred is None:
        return cond_pred

    # in feats proceed on CFG mix
    cond_feats = cond_pred.feats  # (N,C)
    uncond_feats = uncond_pred.feats  # (N,C)

    if uncond_mode == "detach":
        uncond_feats = uncond_feats.detach()
    elif uncond_mode == "mirror":
        cond_feats = cond_feats.detach()
    # "none" mode remains as is

    mixed_feats = cond_feats + scale * (cond_feats - uncond_feats)  # (N,C)

    # Use mixed feats create new SparseTensor
    return cond_pred.replace(mixed_feats)




def auto_device_sparse(model_name="slat_flow_model"):
    """Decorator: move SparseTensor inputs to model device and outputs back."""
    import functools

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(pipeline, x_t, *args, **kwargs):
            model_device = next(pipeline.pipe.models[model_name].parameters()).device
            orig_device = x_t.feats.device

            if model_device == orig_device:
                return fn(pipeline, x_t, *args, **kwargs)
            else:
                raise ValueError(f"Model device {model_device} and original device {orig_device} are different.")

            # create new SparseTensor, without reusing any cache (avoiding cross-device caching issues)
            x_t_new = SparseTensor(
                feats=x_t.feats.to(model_device),  # (N, C)
                coords=x_t.coords.to(model_device),  # (N, 4)
            )
            inputs = [x_t_new] + [a.to(model_device) for a in args]

            result = fn(pipeline, *inputs, **kwargs)

            # Also creates a brand new one when returning SparseTensor
            return SparseTensor(
                feats=result.feats.to(orig_device),  # (N, C)
                coords=result.coords.to(orig_device),  # (N, 4)
            )

        return wrapper
    return decorator


def auto_device_dense(model_name="sparse_structure_flow_model"):
    """Decorator: same as auto_device_sparse but for dense tensors."""
    import functools

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(pipeline, x_t, *args, **kwargs):
            model_device = next(pipeline.pipe.models[model_name].parameters()).device
            orig_device = x_t.device

            if model_device == orig_device:
                return fn(pipeline, x_t, *args, **kwargs)
            else:
                raise ValueError(f"Model device {model_device} and original device {orig_device} are different.")

            x_t_new = x_t.to(model_device)
            inputs = [x_t_new] + [a.to(model_device) for a in args]
            result = fn(pipeline, *inputs, **kwargs)
            return result.to(orig_device)

        return wrapper
    return decorator


@auto_device_sparse("slat_flow_model")
def _predict_sparse_cond_velocity(pipeline, x_t, t_batch, cond_emb):
    """Conditional sparse velocity prediction (auto device)."""
    # ★ Truncate input gradients: avoid multiple steps rollout Concatenation of gradients leads to explosion/disappear
    x_t = x_t.replace(x_t.feats.detach())  # SparseTensor(feats: (N, C), coords: (N, 4))
    return pipeline.sparse.sampling_step(x_t, t_batch, cond_emb, None, 0.0)


def predict_sparse_velocity_with_cfg(
    pipeline,
    x_t: SparseTensor,
    t_val: float,
    cond_emb: torch.Tensor,
    uncond_emb: Optional[torch.Tensor],
    slat_guidance: float,
    cfg_min: float,
    cfg_max: float,
    device: torch.device,
) -> SparseTensor:
    """Sparse velocity + CFG mix. Enable gradient checkpointing at the model level if needed."""
    B = cond_emb.shape[0]
    t_batch = torch.full((B,), t_val, device=device, dtype=torch.float32)  # (B,)
    use_cfg = cfg_min <= t_val <= cfg_max

    # conditional prediction (checkpointing from within the model block processing)
    cond_pred = _predict_sparse_cond_velocity(pipeline, x_t, t_batch, cond_emb)  # SparseTensor

    # CFG mix
    if use_cfg and uncond_emb is not None:
        with torch.no_grad():
            uncond_pred = _predict_sparse_cond_velocity(pipeline, x_t, t_batch, uncond_emb)  # SparseTensor
        return mix_cfg_sparse(cond_pred, uncond_pred, slat_guidance, uncond_mode="detach")  # SparseTensor

    return cond_pred


# =====================================================================
# Generic helper function
# =====================================================================

def prepare_embeddings(
    state: Any,
    device: torch.device,
    cond: Optional[torch.Tensor] = None,
    uncond: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Extract / move cond & uncond embeddings; optional external overrides."""
    if cond is not None:
        cond_emb = cond.to(device)
        uncond_emb = uncond.to(device) if uncond is not None else None
    else:
        cond_emb, uncond_emb = state.extract_embeddings()
        cond_emb = cond_emb.to(device)
        uncond_emb = uncond_emb.to(device) if uncond_emb is not None else None
    return cond_emb, uncond_emb


# =====================================================================
# Dense (Stage 1) Velocity Helper function
# =====================================================================

def _expand_cond_to_batch(
    emb: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    """Expand encoding from (1,...) to (B,...); return as-is if already matched."""
    if emb.shape[0] == 1 and batch_size > 1:
        return emb.repeat(batch_size, *([1] * (emb.dim() - 1)))
    return emb


@auto_device_dense("sparse_structure_flow_model")
def _predict_dense_cond_velocity(
    pipeline: Any,
    x_t: torch.Tensor,
    t_val: float,
    cond_emb: torch.Tensor,
) -> torch.Tensor:
    """Dense conditional velocity. Aligns with sparse: truncate input grads to avoid multi-step graph blow-up."""
    x_t = x_t.detach()
    return pipeline.dense.sampling_step(x_t, t_val, cond_emb, None, 0.0)


def predict_dense_velocity_with_cfg(
    pipeline: Any,
    x_t: torch.Tensor,
    t_val: float,
    cond_emb: torch.Tensor,
    uncond_emb: Optional[torch.Tensor],
    ss_guidance: float,
    cfg_min: float,
    cfg_max: float,
    device: torch.device,
) -> torch.Tensor:
    """Dense velocity + CFG mix (aligned with sparse API)."""
    B = x_t.shape[0]
    use_cfg = cfg_min <= t_val <= cfg_max

    cond_input = _expand_cond_to_batch(cond_emb, B)
    cond_v = _predict_dense_cond_velocity(pipeline, x_t, t_val, cond_input)  # (B, C, R, R, R)

    if use_cfg and uncond_emb is not None:
        uncond_input = _expand_cond_to_batch(uncond_emb, B)
        with torch.no_grad():
            uncond_v = _predict_dense_cond_velocity(pipeline, x_t, t_val, uncond_input)
        return (1 + ss_guidance) * cond_v - ss_guidance * uncond_v.detach()  # (B, C, R, R, R)

    return cond_v
