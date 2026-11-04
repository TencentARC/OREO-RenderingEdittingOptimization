"""
FlowEdit rollout — differential dual-branch sampling in 3D latent space (Dense / Sparse).

- Dual-branch CFG (src reverse / tgt forward)
- Aligned noise updates
- Differential Euler steps

Intended for inference / evaluation (not the contrastive training path).
"""

from typing import Optional, Any
import torch
from tqdm import tqdm
from accelerate import Accelerator

from trellis.modules.sparse import SparseTensor

from .base import (
    _predict_sparse_cond_velocity,
    _predict_dense_cond_velocity,
    _expand_cond_to_batch,
    prepare_embeddings,
)


TrellisState = Any
System = Any


# =====================================================================
# FlowEdit Sparse — Stage 2 (slat_flow_model)
# =====================================================================

def rollout_sparse_flowedit(
    state: TrellisState,
    cfg,
    system: System,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> None:
    """
    Sparse FlowEdit on SLAT Stage 2.
    
    Flow: Euler -> x_src -> dual-branch CFG + aligned noise -> z_edit -> denorm -> state.stage2.z0
    """
    pipeline = system.pipeline
    slat_steps, slat_guidance, slat_rescale_t, cfg_min, cfg_max, _ = pipeline.sparse.get_runtime_params()

    cond_emb, uncond_emb = prepare_embeddings(state, device)
    assert state.coords is not None, "state.coords Missing"
    assert generator is not None, "generator Must be provided by the caller"

    # ---- 1. Standard Euler Rollout → x_src ----
    from .ode import rollout_sparse
    rollout_sparse(state, cfg, system, device, generator=generator, is_training=False)

    # reverse normalize(ode rollout The output is denorm yes, need re-normalize)
    norm = pipeline.pipe.slat_normalization
    std = torch.tensor(norm['std'])[None].to(device)   # (1, C)
    mean = torch.tensor(norm['mean'])[None].to(device)  # (1, C)
    x_src_feats = (state.stage2.z0.feats - mean) / std  # (N, C) normalized
    x_src = state.stage2.z0.replace(x_src_feats)

    # ---- 2. FlowEdit parameters ----
    fe_cfg = cfg.rollout.flowedit
    fe_steps = int(fe_cfg.steps)
    n_max = int(fe_cfg.n_max)
    cfg_scale_tgt = float(fe_cfg.cfg_scale_tgt)
    cfg_scale_src = float(fe_cfg.cfg_scale_src)

    # ---- 3. initialization FlowEdit Status ----
    z_edit_feats = x_src.feats.clone()  # (N, C)
    noise = torch.randn(
        x_src.feats.shape, generator=generator,
        device=device, dtype=x_src.feats.dtype,
    )  # (N, C)

    # ---- 4. Scheduler Configuration ----
    scheduler = pipeline.sparse.scheduler()
    scheduler.set_timesteps(fe_steps, device=device, rescale_t=slat_rescale_t)

    # ---- 5. FlowEdit differential loop ----
    steps = list(scheduler.timesteps)[:-1]
    num_steps = len(steps)
    B = cond_emb.shape[0]

    steps_iter = tqdm(steps, desc="FlowEdit-Sparse", leave=False,
                      disable=not Accelerator().is_main_process)

    with torch.no_grad():
        for i, t in enumerate(steps_iter):
            # Only edit the last n_max steps (earlier steps kept as identity).
            if num_steps - i > n_max:
                continue

            t_val = float(t.item())
            t_prev_val = float(scheduler.timesteps[i + 1].item())
            dt = t_prev_val - t_val  # < 0
            t_batch = torch.full((B,), t_val, device=device, dtype=torch.float32)

            # ---- Source Branch(Reverse CFG)----
            latents_src_feats = (1 - t_val) * x_src.feats + t_val * noise  # (N, C)
            latents_src = x_src.replace(latents_src_feats)

            v_cond_src = _predict_sparse_cond_velocity(
                pipeline, latents_src, t_batch, cond_emb
            )
            v_uncond_src = _predict_sparse_cond_velocity(
                pipeline, latents_src, t_batch, uncond_emb
            )

            # Trellis CFG: (1 + scale) * cond - scale * uncond
            v_cfg_src_feats = (
                (1 + cfg_scale_src) * v_cond_src.feats - cfg_scale_src * v_uncond_src.feats
            )

            # ---- Target Branch(forward CFG)----
            latents_tgt_feats = z_edit_feats + (latents_src_feats - x_src.feats)  # (N, C)
            latents_tgt = x_src.replace(latents_tgt_feats)

            v_cond_tgt = _predict_sparse_cond_velocity(
                pipeline, latents_tgt, t_batch, cond_emb
            )
            v_uncond_tgt = _predict_sparse_cond_velocity(
                pipeline, latents_tgt, t_batch, uncond_emb
            )

            # Trellis CFG: (1 + scale) * cond - scale * uncond
            v_cfg_tgt_feats = (
                (1 + cfg_scale_tgt) * v_cond_tgt.feats - cfg_scale_tgt * v_uncond_tgt.feats
            )

            # ---- difference Euler step ----
            v_delta = v_cfg_tgt_feats - v_cfg_src_feats  # (N, C)
            z_edit_feats = z_edit_feats + dt * v_delta

            # ---- Aligned noise update ----
            noise = noise - (v_cond_tgt.feats - v_uncond_tgt.feats) * (1.0 - t_val)

    # ---- 6. denormalization + mount ----
    denorm_feats = z_edit_feats * std + mean  # (N, C)
    state.stage2.z0 = x_src.replace(denorm_feats)


# =====================================================================
# FlowEdit Dense — Stage 1 (sparse_structure_flow_model)
# =====================================================================

def rollout_dense_flowedit(
    state: TrellisState,
    cfg,
    system: System,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> None:
    """
    Dense FlowEdit on Stage 1 (sparse_structure_flow_model).
    
    Same differential pattern as sparse, operating on dense (B,C,R,R,R) latents.
    """
    pipeline = system.pipeline
    ss_steps, ss_guidance, ss_rescale_t, ss_cfg_min, ss_cfg_max = pipeline.dense.get_runtime_params()

    cond_emb, uncond_emb = prepare_embeddings(state, device)
    assert generator is not None, "generator Must be provided by the caller"

    # ---- 1. Standard Euler Rollout → x_src ----
    from .ode import rollout_dense
    rollout_dense(state, cfg, system, device, generator=generator)
    x_src = state.stage1.z0.clone()  # (B, C, R, R, R)

    # ---- 2. FlowEdit parameters ----
    fe_cfg = cfg.rollout.flowedit
    fe_steps = int(fe_cfg.steps)
    n_max = int(fe_cfg.n_max)
    cfg_scale_tgt = float(fe_cfg.cfg_scale_tgt)
    cfg_scale_src = float(fe_cfg.cfg_scale_src)

    # ---- 3. initialization FlowEdit Status ----
    z_edit = x_src.clone()  # (B, C, R, R, R)
    # ODE init_latents need CPU generator;FlowEdit noise need CUDA generator.
    # from incoming generator Extract seeds and rebuild from CUDA generator, to avoid device type conflicts.
    _fe_seed = generator.initial_seed() + 1 if generator is not None else 0
    cuda_gen = torch.Generator(device=device).manual_seed(_fe_seed)
    noise = torch.randn(
        x_src.shape, generator=cuda_gen,
        device=device, dtype=x_src.dtype,
    )  # (B, C, R, R, R)

    # ---- 4. time step sequence ----
    _, t_pairs = pipeline.dense.scheduler(fe_steps, ss_rescale_t)
    num_steps = len(t_pairs)

    B = x_src.shape[0]

    # ---- 5. FlowEdit differential loop ----
    with torch.no_grad():
        for i, (t, t_prev) in enumerate(tqdm(
            t_pairs, desc="FlowEdit-Dense", leave=False,
            disable=not Accelerator().is_main_process
        )):
            # Only edit the last n_max steps (earlier steps kept as identity).
            if num_steps - i > n_max:
                continue

            t_val = float(t)
            dt = float(t_prev) - t_val  # < 0

            # ---- Source Branch(Reverse CFG)----
            latents_src = (1 - t_val) * x_src + t_val * noise  # (B, C, R, R, R)

            cond_input = _expand_cond_to_batch(cond_emb, B)
            uncond_input = _expand_cond_to_batch(uncond_emb, B)

            v_cond_src = _predict_dense_cond_velocity(pipeline, latents_src, t_val, cond_input)
            v_uncond_src = _predict_dense_cond_velocity(pipeline, latents_src, t_val, uncond_input)

            # Trellis CFG: (1 + scale) * cond - scale * uncond
            v_cfg_src = (1 + cfg_scale_src) * v_cond_src - cfg_scale_src * v_uncond_src

            # ---- Target Branch(forward CFG)----
            latents_tgt = z_edit + (latents_src - x_src)  # (B, C, R, R, R)

            v_cond_tgt = _predict_dense_cond_velocity(pipeline, latents_tgt, t_val, cond_input)
            v_uncond_tgt = _predict_dense_cond_velocity(pipeline, latents_tgt, t_val, uncond_input)

            # Trellis CFG: (1 + scale) * cond - scale * uncond
            v_cfg_tgt = (1 + cfg_scale_tgt) * v_cond_tgt - cfg_scale_tgt * v_uncond_tgt

            # ---- difference Euler step ----
            v_delta = v_cfg_tgt - v_cfg_src  # (B, C, R, R, R)
            z_edit = z_edit + dt * v_delta

            # ---- Aligned noise update ----
            noise = noise - (v_cond_tgt - v_uncond_tgt) * (1.0 - t_val)

    # ---- 6. mount to state(none normalization)----
    state.stage1.z0 = z_edit
