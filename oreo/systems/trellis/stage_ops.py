"""
Trellis v1 StageOps Specific implementation — single stage (SLAT Flow Model).

ABC (StageOps) defined in oreo.systems.utils.stage_ops,
This document provides Trellis v1 Specific implementation to make it accessible autograd_template Arrange templates.

Implementation:
  TrellisOps      — Universal single stage, based on cfg.renderer.type Distribute mesh/gs rendering

Design principles:
  - FlowEdit/Contrastive Ops Reuse TrellisOps of rollout,render and VJP infrastructure
  - vjp_loop Manual merge reg_grads(Adapt 3-sub-step Phase 2)
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any, Dict, Optional

import numpy as np
import torch
from PIL import Image

from oreo.guidance.pipelines.utils.loss_functions import contrastive_loss_step

# ABC from utils Import (model-independent abstraction layer)
from oreo.systems.utils.stage_ops import StageOps  # noqa: F401 — re-export

# Phase function & rendering
from oreo.systems.trellis.forward import decode_and_render_mesh, decode_and_render_gs
from oreo.systems.trellis.phases import phase3_rollout_grad_backward
from oreo.generators.trellis.rollout import rollout_sparse, rollout_dense, RolloutTracker
from oreo.generators.trellis.rollout.base import (
    predict_sparse_velocity_with_cfg,
    prepare_embeddings,
    predict_dense_velocity_with_cfg,
)


# =====================================================================
# Generic implementation — According to cfg.renderer.type Distribute
# =====================================================================

class TrellisOps(StageOps):
    """
    Trellis v1 single stage Ops — SLAT Flow Model.

    Calculation chain:
      dense_sampling → rollout_sparse → decode_and_render_{mesh|gs} → guidance → VJP

    Phase 2 keep 3-sub-step Video memory optimization:
      P2a: no_grad decode → detached comp_rgb
      P2b: guidance-only backward → rgb_grad
      P2c: with-grad decode → backward(rgb_grad) → cond_proxy.grad(only guidance)

    Phase 3 Manual merge reg_grads:
      v_grad = guidance_grad + reg_weight * reg_grad → VJP
    """

    # ═══════════════════════════════════════════════════════
    # Configure query
    # ═══════════════════════════════════════════════════════

    def get_model(self, system):
        """Return DDP packaged slat_flow_model."""
        return system.pipeline.pipe.models['slat_flow_model']

    def get_stage_name(self) -> str:
        return "slat"

    def get_reg_weight(self, system) -> float:
        return system.cfg.train.loss.reg

    def get_guidance_weight(self, system) -> float:
        return system.cfg.train.loss.guidance

    def get_guidance_cfg(self, system):
        return system.cfg.train.guidance

    # ═══════════════════════════════════════════════════════
    # Async Friendly inquiry
    # ═══════════════════════════════════════════════════════

    def get_latent(self, state):
        return state.stage2.z0

    # get_shape_cond → Inherit default None(Single model has no shape cond)

    def decode_render_dict(self, state, system) -> Dict[str, Any]:
        """
        According to cfg.renderer.type distributed to mesh or gs render.

        FlowEdit/Contrastive Subclasses reuse this method, press retained config Select a renderer.
        """
        latents = state.stage2.z0
        device = system.accelerator.device

        renderer_type = system.cfg.renderer.type
        renderer = system.renderers[renderer_type]  # from renderers dict Find
        if renderer_type == "gs":
            return decode_and_render_gs(
                latents, state.cameras,
                system.pipeline, renderer, device,
            )
        else:
            render_out = decode_and_render_mesh(
                latents, state.cameras,
                system.pipeline, renderer, device,
            )
            render_out["color"] = render_out["normal"]
            return render_out

    # ═══════════════════════════════════════════════════════
    # Phase function
    # ═══════════════════════════════════════════════════════

    def pre_rollout(self, state, system, global_step) -> None:
        """Phase 0: Dense Sampling → padding state.coords.

        According to cfg.train.rollout_mode Choose student / pretrained dense rollout.
        If student produce empty coords(decode threshold > 0 All for False),
        automatic fallback Arrive pretrained teacher co-located state.dense_fallback = True.
        """
        pipeline = system.pipeline
        cfg = system.cfg
        device = system.accelerator.device
        generator = torch.Generator(device="cpu").manual_seed(int(cfg.seed))

        rollout_mode = str(cfg.train.rollout_mode)
        if rollout_mode == "pretrained":
            ctx = system.strategy.dense_teacher_context()
        elif rollout_mode == "student":
            ctx = contextlib.nullcontext()
        else:
            raise ValueError(f"Unknown rollout_mode: {rollout_mode!r}")

        with ctx, torch.no_grad():
            rollout_dense(state, cfg, system, device, generator=generator)

        batch_size = state.stage1.z0.shape[0]
        coords = pipeline.dense.decode_to_coords(
            state.stage1.z0, batch_size=batch_size,
        )

        # ── Fallback: student empty coords → teacher ──
        state.dense_fallback = False
        if coords.numel() == 0 and rollout_mode == "student":
            logging.warning(
                "[pre_rollout] Student dense rollout produced empty coords, "
                "falling back to pretrained teacher."
            )
            generator = torch.Generator(device="cpu").manual_seed(int(cfg.seed))
            with system.strategy.dense_teacher_context(), torch.no_grad():
                rollout_dense(state, cfg, system, device, generator=generator)
            coords = pipeline.dense.decode_to_coords(
                state.stage1.z0, batch_size=batch_size,
            )
            state.dense_fallback = True

        state.coords = coords

    def rollout(self, state, system, seed) -> RolloutTracker:
        """
        Phase 1: rollout_sparse → proxy chain + tracker.

        Alignment StageOps Signature:Received seed(calculated by orchestration function),
        Created in-house Generator and call rollout_sparse.
        """
        device = system.accelerator.device
        cfg = system.cfg
        generator = torch.Generator(device=device).manual_seed(seed)
        tracker = RolloutTracker()

        rollout_sparse(
            state, cfg, system, device,
            generator=generator,
            is_training=False,   # Model reasoning no_grad
            tracker=tracker,     # ★ record proxy trajectory
        )
        torch.cuda.empty_cache()
        return tracker

    def decode_render(self, state, system) -> torch.Tensor:
        """
        Phase 2a: decode + render → comp_rgb(Including autograd Figure, connected to proxy chain).

        call decode_render_dict + Mount visual data.
        """
        render_out = self.decode_render_dict(state, system)
        comp_rgb = render_out["color"]  # (B, V, H, W, C)
        state.views_generated.image_tensor = comp_rgb.detach()
        return comp_rgb

    def vjp_loop(self, state, system, tracker: RolloutTracker) -> Dict[str, Any]:
        """
        Phase 3: VJP → θ.grad Accumulate(Delegate phases.phase3_rollout_grad_backward).

        ★ adapt 3-sub-step Phase 2:cond_proxy.grad Contains only guidance gradient,
        phase3_rollout_grad_backward Internal manual merge reg_weight * reg_grads[i].
        """
        reg_weight = self.get_reg_weight(system)
        log = tracker.collect_log(reg_weight=reg_weight)  # loss/reg + grad_norm/*(VJP collected before)

        phase3_rollout_grad_backward(
            state, system, system.cfg, system.accelerator.device, tracker,
        )

        return log


# =====================================================================
# FlowEdit training — Pretrained Rollout + Finetuned Single step denoising
# =====================================================================

class TrellisFlowEditOps(TrellisOps):
    """
    FlowEdit Ops:Rollout + Finetuned Single step denoising + 2D FlowEdit Guidance.

    Training process:
      1. pre_rollout()           — Dense Sampling(Reuse TrellisOps)
      2. rollout()               — no_grad complete rollout → z₀(pretrained or student)
      3. add_noise(z₀, t)       — random sampling t,flow matching Noise → zₜ(normalized domain)
      4. finetune_denoise(zₜ,t) — finetuned model Single step prediction speed → ẑ₀(with gradient)
      5. denormalize(ẑ₀)        — denormalize to decoder input space
      6. decode_render_dict()    — decoder → rendering comp_rgb(Reuse TrellisOps)

    No need VJP,proxy chain,no_sync hack, standard autograd That's it.

    Gradient propagation path:
      loss → rgb_grad → decoder(frozen, There is a calculation diagram) → ẑ₀ → finetune_denoise → θ.grad

    Configuration requirements (cfg.train (below):
      rollout_mode:  "pretrained" | "student"(P1 which model to use rollout)
      noise.t_min:   Time step sampling lower bound (default 0.02)
      noise.t_max:   Time step sampling upper bound (default 0.98)
    """

    # ═══════════════════════════════════════════════════════
    # Rollout(according to cfg.train.rollout_mode Select model)
    # ═══════════════════════════════════════════════════════

    def rollout(self, state, system, seed) -> None:
        """
        complete rollout → clean z₀,no_grad.

        According to cfg.train.rollout_mode Choose which model to use:
          - "pretrained":sparse_teacher_context(), use pretrained weight(off-policy)
          - "student": Directly use the current finetuned weight(on-policy)

        completely no_grad, no need for any proxy chain.

        Side Effects:
            - state.stage2.z0: mount rollout Output SparseTensor(after denormalization)
        """
        device = system.accelerator.device
        cfg = system.cfg
        generator = torch.Generator(device=device).manual_seed(seed)

        rollout_mode = str(cfg.train.rollout_mode)
        if rollout_mode == "pretrained":
            ctx = system.strategy.sparse_teacher_context()
        elif rollout_mode == "student":
            ctx = contextlib.nullcontext()
        else:
            raise ValueError(f"Unknown rollout_mode: {rollout_mode!r}, expected 'pretrained' or 'student'")

        with ctx, torch.no_grad():
            rollout_sparse(
                state, cfg, system, device,
                generator=generator,
                is_training=False,
                tracker=None,  # No need proxy chain
            )
        # rollout_sparse Denormalization has been done and mounted to state.stage2.z0
        torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════════
    # Add noise:z₀ → zₜ(Normalized domain operation)
    # ═══════════════════════════════════════════════════════

    def normalize_latent(self, state, system) -> torch.Tensor:
        """
        will state.stage2.z0.feats From the anti-normalization domain → Normalization domain.

        rollout_sparse Output slat has been denormalized (denorm_feats = feats * std + mean),
        Noise/Denoising needs to be performed in the normalized domain (consistent with training).

        Returns:
            normalized_feats: (N, C) Normalized features
        """
        norm = system.pipeline.pipe.slat_normalization
        device = system.accelerator.device
        std = torch.tensor(norm['std'])[None].to(device)   # (1, C)
        mean = torch.tensor(norm['mean'])[None].to(device)  # (1, C)
        denorm_feats = state.stage2.z0.feats  # (N, C)
        return (denorm_feats - mean) / std  # (N, C)

    def add_noise(self, z0_feats: torch.Tensor, t: float, generator=None) -> torch.Tensor:
        """
        Flow matching Add noise:zₜ = (1-t) * z₀ + t * ε

        Args:
            z0_feats: (N, C) normalized domain clean features
            t: scalar time step ∈ (0, 1)
            generator: random number generator

        Returns:
            zt_feats: (N, C) Features after adding noise
        """
        noise = torch.randn_like(z0_feats)
        zt = (1.0 - t) * z0_feats + t * noise
        return zt

    def sample_timestep(self, system) -> float:
        """
        from inference scheduler Randomly sample a time step from the sequence of time steps.

        Use scheduler The actual time step of (after mu-shift and rescale_t),
        rather than simple Uniform(t_min, t_max), ensuring alignment inference distribution.

        Returns:
            t_val: Sampled time step value (range [0, 1], with scheduler.timesteps consistent)
        """
        pipeline = system.pipeline
        device = system.accelerator.device
        slat_steps, _, slat_rescale_t, _, _, _ = pipeline.sparse.get_runtime_params()

        scheduler = pipeline.sparse.scheduler()
        scheduler.set_timesteps(slat_steps, device=device, rescale_t=slat_rescale_t)
        timesteps = scheduler.timesteps  # Tensor, Arrange from largest to smallest, [0, 1] range

        # remove the last one (usually close to 0)
        timesteps = timesteps[:-1]

        # Optional: Limit the sampling range (t_min / t_max in config already in [0,1] range)
        cfg = system.cfg
        t_min = float(cfg.train.noise.get("t_min", 0.02))
        t_max = float(cfg.train.noise.get("t_max", 0.98))
        mask = (timesteps >= t_min) & (timesteps <= t_max)
        valid_timesteps = timesteps[mask]
        if len(valid_timesteps) == 0:
            valid_timesteps = timesteps  # fallback

        # Choose one at random
        idx = torch.randint(0, len(valid_timesteps), (1,)).item()
        return float(valid_timesteps[idx].item())

    # ═══════════════════════════════════════════════════════
    # Velocity predict(student / teacher)
    # ═══════════════════════════════════════════════════════

    def _predict_velocity_impl(
        self, state, system, zt_feats: torch.Tensor, t_val: float,
        cond: Optional[torch.Tensor] = None,
        uncond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Internal Sharing: Build SparseTensor input + predict_sparse_velocity_with_cfg → v_feats.

        The caller decides whether the sparse_teacher_context / no_grad Call next.

        Args:
            state: TrellisState(Required state.stage2.z0 provide coords)
            system: TrellisSystem
            zt_feats: (N, C) Normalized domain features
            t_val: time step value ([0, 1000] range)
            cond: (B, S, C) Conditional encoding (optional, if not passed, follow state read)
            uncond: (B, S, C) Unconditional encoding (optional, if not passed, it will be state read)

        Returns:
            v_feats: (N, C) Predicted speed characteristics
        """
        pipeline = system.pipeline
        device = system.accelerator.device
        _, slat_guidance, _, slat_cfg_min, slat_cfg_max, _ = pipeline.sparse.get_runtime_params()

        # build SparseTensor input
        slat = state.stage2.z0
        x_t = slat.replace(zt_feats)  # SparseTensor with zt_feats, share coords

        # conditional encoding
        cond_emb, uncond_emb = prepare_embeddings(state, device, cond, uncond)

        velocity = predict_sparse_velocity_with_cfg(
            pipeline, x_t, t_val, cond_emb, uncond_emb,
            slat_guidance, slat_cfg_min, slat_cfg_max, device,
        )  # SparseTensor
        return velocity.feats  # (N, C)

    def predict_velocity_student(self, state, system, zt_feats: torch.Tensor, t_val: float) -> torch.Tensor:
        """
        Finetuned (student) model Speed prediction (with autograd Picture arrived θ).

        Returns:
            v_feats: (N, C) Yes autograd Figure
        """
        return self._predict_velocity_impl(state, system, zt_feats, t_val)

    def predict_velocity_teacher(
        self, state, system, zt_feats: torch.Tensor, t_val: float,
        cond: Optional[torch.Tensor] = None,
        uncond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Pretrained (teacher) model speed prediction (sparse_teacher_context + no_grad,detached).

        Args:
            cond: (B, S, C) Conditional encoding (optional, if not passed, follow state read)
            uncond: (B, S, C) Unconditional encoding (optional, if not passed, it will be state read)

        Returns:
            v_feats: (N, C) detached
        """
        with system.strategy.sparse_teacher_context(), torch.no_grad():
            v = self._predict_velocity_impl(state, system, zt_feats, t_val, cond=cond, uncond=uncond)
        return v.detach()

    def denormalize_feats(self, feats: torch.Tensor, system) -> torch.Tensor:
        """
        Denormalization: Normalize domain features → decoder Enter the domain.

        Args:
            feats: (N, C) Normalized domain features
            system: TrellisSystem

        Returns:
            denorm_feats: (N, C) Features after denormalization
        """
        norm = system.pipeline.pipe.slat_normalization
        device = system.accelerator.device
        std = torch.tensor(norm['std'])[None].to(device)   # (1, C)
        mean = torch.tensor(norm['mean'])[None].to(device)  # (1, C)
        return feats * std + mean  # (N, C)


# =====================================================================
# Contrastive FlowEdit Ops — in latent Comparative learning in space
# =====================================================================

class TrellisContrastiveOps(TrellisFlowEditOps):
    """
    Contrastive FlowEdit Ops: Expand P4/P5 for latent Spatial contrastive learning.

    inheritance TrellisFlowEditOps of P0–P3.5(dense sampling,pretrained rollout,
    Add noise,velocity prediction,velocity reg).

    New method (P4/P5):
      decode_render_teacher   — rendering Teacher z₀ → src tensor(mounted to state)
      edit_views              — FlowEdit Edit src → tgt tensor(from state read src)
      encode_conditions       — tensor→PIL→preprocess→DINOv2 encoding → c_src, c_tgt
      predict_x0_teacher_with_cond — Teacher Single step denoising with custom condition
      contrastive_loss        — Contrast loss(latent space)

    Gradient propagation path:
      contrastive_loss → z0_hat_norm → v_proxy → (relay) → v_student → θ.grad
    """

    def decode_render_teacher(self, state, system) -> None:
        """
        P4a: rendering Teacher z₀ → src image tensor.

        Preconditions: state.stage2.z0 == z0_teacher(P1 has not been modified since the end).

        Side Effects:
            - state.views_generated.image_tensor: (B, V, H, W, C) detached
        """
        state.stage2.z0._spatial_cache.clear()
        torch.cuda.empty_cache()

        with torch.no_grad():
            render_out = self.decode_render_dict(state, system)
        state.views_generated.image_tensor = render_out["color"].detach()  # (B, V, H, W, C)

        state.stage2.z0._spatial_cache.clear()
        del render_out
        torch.cuda.empty_cache()

    def edit_views(self, state, system) -> None:
        """
        P4b: FlowEdit Edit src → tgt.

        from state.views_generated.image_tensor read src,
        Use system.guidance.compute_guidance do 2D FlowEdit edit,
        The result is unified as (B, V, H, W, C) mount to state.views_edited.

        Side Effects:
            - state.views_edited.image_tensor: (B, V, C, H, W) detached
            - state.trackers.guidance:         List[StateTracker]
        """
        src_tensor = state.views_generated.image_tensor  # (B, V, H, W, C)
        guidance_cfg = self.get_guidance_cfg(system)
        accelerator = system.accelerator

        guidance_result = system.guidance.edit(
            src_tensor,
            state.views_conditioned.image_pils,
            guidance_cfg=guidance_cfg,
            rank=accelerator.process_index,
        )

        edited_imgs = guidance_result.edited_imgs  # (B, V, C, H, W)
        state.views_edited.image_tensor = edited_imgs.detach()

        state.trackers.guidance = guidance_result.trackers

        del guidance_result
        torch.cuda.empty_cache()

    def encode_conditions(self, state, system) -> None:
        """
        P4c: tensor → PIL → preprocess → DINOv2 encoding → c_src, c_tgt.

        from state read image_tensor(both (B, V, H, W, C)), take view 0 turn PIL,
        by preprocess_image(rembg Go and carry + bbox crop center + resize 518) after
        by encode_image(DINOv2) encoded as patch token.

        Side Effects:
            - state.views_generated.image_condition: (B, S, C)
            - state.views_edited.image_condition:    (B, S, C)
        """
        pipe = system.pipeline.pipe

        # src: (B, V, H, W, C), tgt: (B, V, C, H, W)
        src_tensor = state.views_generated.image_tensor  # (B, V, H, W, C)
        tgt_tensor = state.views_edited.image_tensor     # (B, V, C, H, W)
        src_pils = [
            Image.fromarray((src_tensor[b, 0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8))
            for b in range(src_tensor.shape[0])
        ]
        tgt_pils = [
            Image.fromarray((tgt_tensor[b, 0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8))
            for b in range(tgt_tensor.shape[0])
        ]

        with torch.no_grad():
            src_proc = [pipe.preprocess_image(img) for img in src_pils]
            tgt_proc = [pipe.preprocess_image(img) for img in tgt_pils]
            c_src = pipe.encode_image(src_proc)  # (B, S, C)
            c_tgt = pipe.encode_image(tgt_proc)  # (B, S, C)

        state.views_generated.image_condition = c_src
        state.views_edited.image_condition = c_tgt

    @staticmethod
    def compute_dino_similarity(state) -> Tuple[Dict[str, float], torch.Tensor, torch.Tensor]:
        """
        Calculate src / tgt with input of DINO cosine similarity.

        read:
            state.views_conditioned.cond_embed       — (B, S, C) input image DINO embedding
            state.views_generated.image_condition     — (B, S, C) src (Teacher rendering)
            state.views_edited.image_condition        — (B, S, C) tgt (FlowEdit Edit)

        Returns:
            log_dict: sim/src_input, sim/tgt_input, sim/tgt_gt_src
            sim_src: (B,) per-sample cosine similarity
            sim_tgt: (B,) per-sample cosine similarity
        """
        c_input = state.views_conditioned.cond_embed       # (B, S, C)
        c_src = state.views_generated.image_condition       # (B, S, C)
        c_tgt = state.views_edited.image_condition          # (B, S, C)

        # mean pool over patch tokens → (B, C)
        e_input = c_input.mean(dim=1)
        e_src = c_src.mean(dim=1)
        e_tgt = c_tgt.mean(dim=1)

        sim_src = torch.nn.functional.cosine_similarity(e_src, e_input, dim=-1)  # (B,)
        sim_tgt = torch.nn.functional.cosine_similarity(e_tgt, e_input, dim=-1)  # (B,)

        log_dict = {
            "sim/src_input": sim_src.mean().item(),
            "sim/tgt_input": sim_tgt.mean().item(),
            "sim/tgt_gt_src": (sim_tgt > sim_src).float().mean().item(),
        }
        return log_dict, sim_src, sim_tgt

    @staticmethod
    def adaptive_swap_conditions(state, sim_src: torch.Tensor, sim_tgt: torch.Tensor) -> float:
        """
        Per-sample Swap c_src / c_tgt: When sim_src > sim_tgt time exchange.

        Modify:
            state.views_generated.image_condition
            state.views_edited.image_condition

        Returns:
            swap_rate: exchanged sample Proportion
        """
        swap_mask = sim_src > sim_tgt  # (B,)
        swap_rate = swap_mask.float().mean().item()

        if swap_mask.any():
            c_src = state.views_generated.image_condition  # (B, S, C)
            c_tgt = state.views_edited.image_condition     # (B, S, C)
            mask = swap_mask[:, None, None]                # (B, 1, 1)
            new_src = torch.where(mask, c_tgt, c_src)
            new_tgt = torch.where(mask, c_src, c_tgt)
            state.views_generated.image_condition = new_src
            state.views_edited.image_condition = new_tgt

        return swap_rate

    def predict_x0_teacher_with_cond(
        self,
        state,
        system,
        zt_feats: torch.Tensor,
        t_val: float,
        cond_source: str = "edited",
    ) -> torch.Tensor:
        """
        P5a/b: Teacher Single step denoising with condition from state → ẑ₀.

        temporary replacement state Conditional coding in, call predict_velocity_teacher,
        Then restore the original condition.

        Args:
            zt_feats: (N, C) Noise feature (detached)
            t_val: time step
            cond_source: from state which subcontainer reads image_condition.
                         "edited"    → state.views_edited.image_condition
                         "generated" → state.views_generated.image_condition

        Returns:
            x0: (N, C) detached
        """
        if cond_source == "edited":
            cond_emb = state.views_edited.image_condition
        elif cond_source == "generated":
            cond_emb = state.views_generated.image_condition
        else:
            raise ValueError(f"Unknown cond_source: {cond_source!r}, expected 'edited' or 'generated'")

        # teacher_cfg=True  → uncond=zeros, go CFG
        # teacher_cfg=False → uncond=None, skip CFG
        teacher_cfg = system.cfg.train.loss.contrastive.teacher_cfg
        uncond_emb = torch.zeros_like(cond_emb) if teacher_cfg else None
        v_teacher = self.predict_velocity_teacher(
            state, system, zt_feats, t_val,
            cond=cond_emb,
            uncond=uncond_emb,
        )
        x0 = zt_feats - t_val * v_teacher  # (N, C), detached
        return x0

    @staticmethod
    def contrastive_loss(
        z0_stu: torch.Tensor,
        z0_tea_tgt: torch.Tensor,
        z0_tea_src: torch.Tensor,
        ada: bool = True,
        eps: float = 1e-4,
    ) -> torch.Tensor:
        """
        P5c: Contrast loss — latent space.

        let student Forecast is close teacher_tgt(positive sample), stay away teacher_src(negative sample).

        Args:
            z0_stu:     (N, C) There is a gradient (via v_proxy)
            z0_tea_tgt: (N, C) detached, positive
            z0_tea_src: (N, C) detached, negative
            ada: adaptive normalization
            eps: ada epsilon

        Returns:
            scalar loss
        """
        return contrastive_loss_step(
            z0_stu.unsqueeze(0),       # (1, N, C)
            z0_tea_tgt.unsqueeze(0),   # (1, N, C)
            z0_tea_src.unsqueeze(0),   # (1, N, C)
            ada=ada,
            eps=eps,
        )


# =====================================================================
# Dense Ops — Stage 1 (sparse_structure_flow_model)
# =====================================================================

class TrellisDenseOps(TrellisFlowEditOps):
    """
    Dense (Stage 1) Ops — sparse_structure_flow_model.

    Operation dense Tensor (B, C, R, R, R) rather than SparseTensor.
    None normalization(Stage 1 does not exist ss_normalization).
    Not available decode_render — Reuse Sparse stage rendering result.
    """

    # ═══════════════════════════════════════════════════════
    # Configure query
    # ═══════════════════════════════════════════════════════

    def get_model(self, system):
        """Return DDP packaged sparse_structure_flow_model."""
        return system.pipeline.pipe.models['sparse_structure_flow_model']

    def get_stage_name(self) -> str:
        return "ss"

    def get_latent(self, state):
        return state.stage1.z0

    # ═══════════════════════════════════════════════════════
    # Rollout
    # ═══════════════════════════════════════════════════════

    def rollout(self, state, system, seed) -> None:
        """
        Dense rollout → clean z_s,no_grad.

        Side Effects:
            - state.stage1.z0: (B, C, R, R, R) dense latent
        """
        device = system.accelerator.device
        cfg = system.cfg
        generator = torch.Generator(device="cpu").manual_seed(seed)

        rollout_mode = str(cfg.train.rollout_mode)
        if rollout_mode == "pretrained":
            ctx = system.strategy.dense_teacher_context()
        elif rollout_mode == "student":
            ctx = contextlib.nullcontext()
        else:
            raise ValueError(f"Unknown rollout_mode: {rollout_mode!r}")

        with ctx, torch.no_grad():
            rollout_dense(
                state, cfg, system, device,
                generator=generator,
            )
        torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════════
    # Normalize / Denormalize — Stage 1 None normalization
    # ═══════════════════════════════════════════════════════

    def normalize_latent(self, state, system) -> torch.Tensor:
        """Stage 1 None normalization, return directly z0."""
        return state.stage1.z0  # (B, C, R, R, R)

    def denormalize_feats(self, feats: torch.Tensor, system) -> torch.Tensor:
        """Stage 1 None normalization, identity mapping."""
        return feats

    # ═══════════════════════════════════════════════════════
    # Noise / Sampling time step
    # ═══════════════════════════════════════════════════════

    def sample_timestep(self, system) -> float:
        """
        from Stage 1 of scheduler Randomly sampled from a sequence of time steps.
        """
        pipeline = system.pipeline
        ss_steps, _, ss_rescale_t, _, _ = pipeline.dense.get_runtime_params()

        t_seq, _ = pipeline.dense.scheduler(ss_steps, ss_rescale_t)
        # Remove the last one (closer to 0)
        t_seq = t_seq[:-1]

        cfg = system.cfg
        t_min = float(cfg.train.noise.t_min)
        t_max = float(cfg.train.noise.t_max)
        valid = [t for t in t_seq if t_min <= t <= t_max]
        if not valid:
            valid = list(t_seq)

        idx = torch.randint(0, len(valid), (1,)).item()
        return float(valid[idx])

    # ═══════════════════════════════════════════════════════
    # Velocity Forecast — dense Tensor
    # ═══════════════════════════════════════════════════════

    def _predict_velocity_impl(
        self, state, system, zt: torch.Tensor, t_val: float,
        cond: Optional[torch.Tensor] = None,
        uncond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Dense velocity Forecast with CFG.

        Args:
            zt: (B, C, R, R, R)
            t_val: time step ∈ [0, 1]

        Returns:
            velocity: (B, C, R, R, R)
        """
        pipeline = system.pipeline
        device = system.accelerator.device
        _, ss_guidance, _, ss_cfg_min, ss_cfg_max = pipeline.dense.get_runtime_params()

        cond_emb, uncond_emb = prepare_embeddings(state, device, cond, uncond)

        return predict_dense_velocity_with_cfg(
            pipeline, zt, t_val, cond_emb, uncond_emb,
            ss_guidance, ss_cfg_min, ss_cfg_max, device,
        )  # (B, C, R, R, R)

    def predict_velocity_student(self, state, system, zt: torch.Tensor, t_val: float) -> torch.Tensor:
        return self._predict_velocity_impl(state, system, zt, t_val)

    def predict_velocity_teacher(
        self, state, system, zt: torch.Tensor, t_val: float,
        cond: Optional[torch.Tensor] = None,
        uncond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        with system.strategy.dense_teacher_context(), torch.no_grad():
            v = self._predict_velocity_impl(state, system, zt, t_val, cond=cond, uncond=uncond)
        return v.detach()

    # ═══════════════════════════════════════════════════════
    # Not available decode / render
    # ═══════════════════════════════════════════════════════

    def decode_render_dict(self, state, system):
        raise NotImplementedError("TrellisDenseOps Not available decode_render, should be reused Sparse stage Rendering results.")


# =====================================================================
# Dense Contrastive Ops — in Dense latent Comparative learning in space
# =====================================================================

class TrellisDenseContrastiveOps(TrellisDenseOps):
    """
    Dense Contrastive Ops: in Dense (B,C,R,R,R) latent Space for comparative learning.

    Reuse Sparse stage of c_src / c_tgt(DINOv2 conditional encoding),
    Does not render itself/Edit/Encoding.

    Method source:
      rollout / normalize / add_noise / sample_timestep / velocity → TrellisDenseOps
      predict_x0_teacher_with_cond → This category (dense version)
      contrastive_loss → TrellisContrastiveOps(reuse)
    """

    def predict_x0_teacher_with_cond(
        self,
        state,
        system,
        zt: torch.Tensor,
        t_val: float,
        cond_source: str = "edited",
    ) -> torch.Tensor:
        """
        Teacher Single step denoising with condition → ẑ₀ (dense).

        Args:
            zt: (B, C, R, R, R) Noise latent
            t_val: time step
            cond_source: "edited" | "generated"

        Returns:
            x0: (B, C, R, R, R) detached
        """
        if cond_source == "edited":
            cond_emb = state.views_edited.image_condition
        elif cond_source == "generated":
            cond_emb = state.views_generated.image_condition
        else:
            raise ValueError(f"Unknown cond_source: {cond_source!r}")

        # teacher_cfg=True  → uncond=zeros, go CFG
        # teacher_cfg=False → uncond=None, skip CFG
        teacher_cfg = system.cfg.train.loss.contrastive.teacher_cfg
        uncond_emb = torch.zeros_like(cond_emb) if teacher_cfg else None
        v_teacher = self.predict_velocity_teacher(
            state, system, zt, t_val,
            cond=cond_emb,
            uncond=uncond_emb,
        )
        return zt - t_val * v_teacher  # (B, C, R, R, R), detached

    @staticmethod
    def contrastive_loss(
        z0_stu: torch.Tensor,
        z0_tea_tgt: torch.Tensor,
        z0_tea_src: torch.Tensor,
        ada: bool = True,
        eps: float = 1e-4,
    ) -> torch.Tensor:
        """
        Dense Contrast loss — in (B, C, R, R, R) space operations.

        will 5D tensor flattened to (B, C*R*R*R) call after contrastive_loss_step.
        """
        B = z0_stu.shape[0]
        return contrastive_loss_step(
            z0_stu.reshape(B, 1, -1),       # (B, 1, C*R³)
            z0_tea_tgt.reshape(B, 1, -1),   # (B, 1, C*R³)
            z0_tea_src.reshape(B, 1, -1),   # (B, 1, C*R³)
            ada=ada,
            eps=eps,
        )
