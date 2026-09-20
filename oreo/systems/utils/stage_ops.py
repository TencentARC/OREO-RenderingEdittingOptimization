"""
StageOps abstract base class — Computational operation protocol for the training phase.

Define each training phase "what to do" interface, does not contain any cleanup/Arrange/Life cycle knowledge.
same one Ops Instances can be synchronized at/asynchronous/single stage/Reuse in multi-stage templates.

This file only contains model-independent ABC and exception classes;
specific implementation (ShapeOps / TexOps etc.) are provided by each model backend.

Usage::

    from oreo.systems.utils.stage_ops import StageOps

    class MyOps(StageOps):
        ...

Design principles:
  - Cleanup strategy consists of Slot(orchestration layer) determines, not by Ops decide
  - Ops Just answer "What is the calculation of this stage itself?"
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict

import torch

if TYPE_CHECKING:
    from oreo.generators.trellis2.rollout import RolloutTracker


# =====================================================================
# Abnormal
# =====================================================================

class StageSkipError(Exception):
    """Stage downgrade exception — The preconditions are not met (such as meshes for None), skip subsequent calculations."""
    pass


# =====================================================================
# abstract base class
# =====================================================================

class StageOps(ABC):
    """
    Computational operation interface for a single training stage.

    Computational logic unique to the packaging stage (rollout / decode+render / VJP),
    Does not include any knowledge of cleanup strategies, lifecycle management, or orchestration.

    Subclasses only need to pay attention to "How do I calculate this at this stage?", no need to know:
    - Am I in the last stage?
    - mine subs/meshes Whether it is needed in subsequent stages
    - Should I execute it synchronously or pipeline it asynchronously?
    """

    # ═══════════════════════════════════════════════════════
    # Configure query
    # ═══════════════════════════════════════════════════════

    @abstractmethod
    def get_model(self, system):
        """Return to this stage model(DDP after packaging)."""
        ...

    @abstractmethod
    def get_stage_name(self) -> str:
        """Return stage name ("shape" / "tex"), used for pipeline.get_stage_config() Wait."""
        ...

    @abstractmethod
    def get_reg_weight(self, system) -> float:
        """Return reg loss weight(cfg.{stage}.train.loss.reg)."""
        ...

    @abstractmethod
    def get_guidance_weight(self, system) -> float:
        """Return guidance loss weight(cfg.{stage}.train.loss.guidance)."""
        ...

    @abstractmethod
    def get_guidance_cfg(self, system):
        """Return guidance configure(cfg.{stage}.guidance)."""
        ...

    # ═══════════════════════════════════════════════════════
    # Async Friendly inquiry (VJP loop / P2-grad multiplexing)
    # ═══════════════════════════════════════════════════════

    @abstractmethod
    def get_latent(self, state):
        """Return to this stage latent(dense / sparse),VJP Pass .replace() build x_t."""
        ...

    def get_shape_cond(self, state):
        """Return VJP required shape_cond.Shape→None, Tex→shape_slat_norm. Default None."""
        return None

    @abstractmethod
    def decode_render_dict(self, state, system) -> Dict[str, Any]:
        """
        decode+render → Original dictionary (including 'color' key).

        with decode_render The difference: don't do vis Mount, don't do it subs/meshes Assignment.
        for asynchronous P2-grad Rerun (just comp_rgb.backward) and P2-no-grad.

        Subclass decode_render This method should be called + additional vis Mount logic.
        """
        ...

    # ═══════════════════════════════════════════════════════
    # Phase function
    # ═══════════════════════════════════════════════════════

    def pre_rollout(self, state, system, global_step) -> None:
        """
        Phase 0: Rollout Preparation (optional).

        - Shape: dense_sampling → padding state.coords
        - Tex (standalone): shape_frozen_prepare → padding shape product + detach
        - Tex (from shape): no-op(shape Products are provided by upstream)

        Default no-op, subclasses override as needed.
        """
        pass

    @abstractmethod
    def rollout(self, state, system, seed) -> RolloutTracker:
        """
        Phase 1: rollout → proxy chain + tracker.

        padding state.features.{shape,tex}_slat (proxy chain).
        Return RolloutTracker(Including input/output trajectory, timesteps, reg_grads).
        """
        ...

    @abstractmethod
    def decode_render(self, state, system) -> torch.Tensor:
        """
        Phase 2a: decode + render → comp_rgb.

        Return comp_rgb (B, V, H, W, 3).
        Is there any when calling autograd The graph depends on the caller context (synchronous templates with gradients, async templates no_grad).

        Side effects:
        - mount vis data arrives state(e.g., views_generated.shape_tensor)
        - Mount the intermediate product to state(e.g., features.subs, features.meshes)

        Raises:
            StageSkipError: The preconditions are not met (such as meshes for None)
        """
        ...

    @abstractmethod
    def vjp_loop(self, state, system, tracker: RolloutTracker) -> Dict[str, Any]:
        """
        Phase 3: VJP → θ.grad Accumulate.

        gradually/Batch recalculation f_θ, use tracker gradient in VJP → Model parameter gradient accumulation.
        Internal self-cleaning tracker data.

        Returns:
            log dictionary (usually containing loss/reg or empty dict)
        """
        ...
