"""
three stages Autograd Universal template — Parameterized synchronized training steps.

The three-stage training steps (P0 + P1 + P2a/P2 + P3) is extracted as a general function,
Pass StageOps Calculations specific to the parameterization phase, via clean_for_vjp Callback parameterized cleanup strategy.

This module does not rely on any specific model backend, only through StageOps agreement with state/system of
duck-typed Interface communication. any implemented StageOps All model backends can be used.

Usage::

    from oreo.systems.utils.autograd_template import three_phase_step
    from my_model.stage_ops import MyShapeOps

    three_phase_step(MyShapeOps(), state, system, ...,
        clean_for_vjp=lambda s: s.prepare_for_vjp())

state / system Implied agreement:
    state.views_conditioned.image_pils  — Conditional image list
    state.attach_guidance_result(result) — mount guidance result
    state.stage2.reg_loss              — reg loss tensor (Optional)
    system.accelerator                  — Accelerate accelerator
    system.guidance.compute_guidance()  — sync guidance forward
    system.cfg.seed                     — global seed
"""

import logging
from typing import Any, Callable, Dict

import torch

from oreo.systems.utils.stage_ops import StageOps, StageSkipError
from oreo.systems.utils.logging import build_autograd_step_log


# =====================================================================
# Phase 2: Universal Guidance + Backward
# =====================================================================

def _phase2_guidance_and_backward(
    ops: StageOps,
    state,
    system,
    comp_rgb: torch.Tensor,
) -> Dict[str, Any]:
    """
    Universal Phase 2(synchronous version): guidance + reg merge backward → padding tracker gradient → Release chart.

    ★ Pass ops get guidance_weight / reg_weight / guidance_cfg,
      eliminate shape/tex two sets phase2_guidance_and_backward of repetition.

    Mathematics:
      total_loss = guidance_loss * w_guid + reg_loss * w_reg
      accelerator.backward(total_loss)
      → output_trajectory[t].grad = ∂total_loss/∂cond_proxy_t

    Args:
        ops: stage Ops(provided guidance configuration)
        state: Training status (including views_conditioned, regularization etc.)
        system: training system (including guidance, accelerator)
        comp_rgb: Phase 2a The resulting rendering (B, V, H, W, 3)(Yes autograd Figure)

    Returns:
        guidance Log dictionary (including loss/guidance, loss/reg etc.)
    """
    accelerator = system.accelerator
    device = accelerator.device
    guidance_weight = ops.get_guidance_weight(system)
    reg_weight = ops.get_reg_weight(system)

    # 1. Guidance Forward (synchronous blocking)
    guidance_result = system.guidance.compute_guidance(
        comp_rgb,
        state.views_conditioned.image_pils,
        guidance_cfg=ops.get_guidance_cfg(system),
        rank=accelerator.process_index,
    )
    state.attach_guidance_result(guidance_result)

    # 2. merge loss: guidance + reg
    # comp_rgb ← renderer ← decoder ← slat ← scheduler ← CFG ← cond_proxy
    # reg_loss ← MSE/velocity ← CFG ← cond_proxy
    # → The two gradients converge to cond_proxy.grad
    total_loss = guidance_result.loss.to(device) * guidance_weight  # ()
    reg_loss = state.stage2.reg_loss
    if reg_loss is not None:
        total_loss = total_loss + reg_weight * reg_loss  # ()

    # 3. Backward(back-transmission all the way to output_trajectory[t].grad)
    accelerator.backward(total_loss)

    # 4. Build log
    guidance_log: Dict[str, Any] = {}
    if guidance_result.loss_dict:
        guidance_log.update({
            f"loss/{k}": v.item()
            for k, v in guidance_result.loss_dict.items()
            if v is not None
        })
    guidance_log["loss/guidance"] = (
        guidance_result.loss.to(device) * guidance_weight
    ).item()
    if reg_loss is not None:
        guidance_log["loss/reg"] = reg_loss.item()

    # 5. Release all computation graph references
    del comp_rgb, total_loss, guidance_result, reg_loss
    state.stage2.reg_loss = None
    torch.cuda.empty_cache()

    return guidance_log


# =====================================================================
# Universal three-phase training steps
# =====================================================================

def three_phase_step(
    ops: StageOps,
    state,
    system,
    global_step: int,
    profiler,
    clean_for_vjp: Callable,
    prefix: str = "",
) -> Dict[str, Any]:
    """
    Universal three stages Autograd training steps.

    Arrangement:
      P0 (pre_rollout) → P1 (rollout + tracker)
      → P2a (decode + render) → P2 (guidance + backward)
      → clean_for_vjp → P3 (VJP) → Return to log

    Args:
        ops: stage-specific calculation operations (any StageOps implementation)
        state: Already attach_batch status
        system: training system
        global_step: global steps
        profiler: PhaseProfiler(tick / collect interface)
        clean_for_vjp: P2→P3 Transition cleanup function (injected by caller, encoding cleanup strategy)
        prefix: profiler tick and log key prefix (such as "shape/" or "tex/")

    Returns:
        Merged log dictionary (excluding profiler Timing——It is up to the caller to decide whether to collect)
    """
    seed = int(system.cfg.seed)

    # ── Phase 0: Prepare(dense_sampling / shape_frozen_prepare / no-op)──
    profiler.tick(f"{prefix}P0_pre_rollout")
    ops.pre_rollout(state, system, global_step)

    # ── Phase 1: Rollout → proxy chain + tracker ──
    profiler.tick(f"{prefix}P1_rollout")
    tracker = ops.rollout(state, system, seed)

    # ── Phase 2a + Phase 2: Decode/Render + Guidance Backward ──
    profiler.tick(f"{prefix}P2a_decode_render")
    comp_rgb = None
    skip_phase3 = False
    guidance_log: Dict[str, Any] = {}
    try:
        comp_rgb = ops.decode_render(state, system)

        profiler.tick(f"{prefix}P2_guidance_backward")
        guidance_log = _phase2_guidance_and_backward(ops, state, system, comp_rgb)
    except (torch.cuda.OutOfMemoryError, StageSkipError) as e:
        # P2a OOM / decode Preconditions are not met → skip P3
        # ★ Don't do it reg-only VJP: Very large sample VJP It may take too long, resulting in NCCL timeout.
        # Security:P2a/P2 Does not go through model parameters and does not trigger DDP hooks, will not cause distributed deadlock.
        logging.warning(
            f"[Step {global_step}] {prefix}P2a/P2 failed: {e} → skip P3"
        )
        skip_phase3 = True
        del comp_rgb
        torch.cuda.empty_cache()

    # ── P2→P3 Transition: caller-injected cleanup strategy ──
    clean_for_vjp(state)

    # ── Phase 3: VJP → θ.grad accumulate ──
    if not skip_phase3:
        profiler.tick(f"{prefix}P3_grad_backward")
        phase3_log = ops.vjp_loop(state, system, tracker)
    else:
        profiler.tick(f"{prefix}P3_skip")
        # clean only tracker data, not executed VJP
        del tracker.input_trajectory[:], tracker.output_trajectory[:]
        del tracker.timesteps[:]
        torch.cuda.empty_cache()
        phase3_log = {}

    profiler.tick(f"{prefix}end")

    # Merge log (excluding profiler Timing——Added on demand by the caller)
    return build_autograd_step_log(
        guidance_log, ops.get_reg_weight(system), phase3_log, prefix=prefix,
    )
