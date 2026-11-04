"""
RolloutTracker / VelocityTracker: cond-level proxies for three-phase autograd.

Phase 1 writes trajectory + precomputes reg grads; Phase 2 fills guidance .grad;
Phase 3 VJP consumes both. No intermediate slat_proxy — decode_render grads flow via tracker.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch


@dataclass
class RolloutTracker:
    """Inter-phase carrier: stores timesteps, input/output trajectories, and reg grads for Phase 3 VJP."""

    # Phase 1 Write:rollout input for each step/Output snapshot + time step
    input_trajectory: List[torch.Tensor] = field(default_factory=list)
    output_trajectory: List[torch.Tensor] = field(default_factory=list)
    #   ★ save cond_pred proxy(CFG before),.grad by Phase 2 guidance backward padding
    timesteps: List[float] = field(default_factory=list)

    # Phase 1 Write:reg gradient (computed in advance, with Phase 2 decoupling)
    reg_grads: List[torch.Tensor] = field(default_factory=list)
    #   ★ pure data (detach),Phase 2 OOM Not lost either
    reg_loss_val: Optional[float] = None
    #   ★ reg loss Scalar value, used for logging

    def collect_log(self, reg_weight: float = 1.0) -> Dict[str, float]:
        """
        collect tracker All logging indicators in .

        in VJP Call before loop (Phase 2c backward Filled .grad,Phase 1 Filled reg_grads).

        Args:
            reg_weight: reg Gradient weights (used to calculate effective gradients norm, align guidance gradient scale).
                        guidance Gradient included guidance_weight(Phase 2b backward embedded),
                        reg The gradient is the original value, which needs to be multiplied by reg_weight Talent and guidance Same scale.
                        Default 1.0(i.e. reporting the original reg gradient norm).

        Returns:
            Log dictionary, which may contain:
            - loss/reg:           reg loss scalar value
            - grad_norm/guidance: guidance Gradient average L2 norm (proxy level, including guidance_weight)
            - grad_norm/reg:     reg Gradient average L2 norm (proxy level, multiply reg_weight after)
            - grad_norm/ratio:   guidance / reg Effective gradient norm ratio
        """
        log: Dict[str, float] = {}

        # ---- loss/reg ----
        if self.reg_loss_val is not None:
            log["loss/reg"] = self.reg_loss_val

        # ---- gradient norm ----
        T = len(self.timesteps)
        has_reg = len(self.reg_grads) == T

        guid_norms: List[float] = []
        reg_norms: List[float] = []

        for i in range(T):
            g = self.output_trajectory[i].grad  # (N, C) or None
            if g is None:
                continue
            guid_norms.append(g.norm().item())
            if has_reg and reg_weight != 0:
                # ★ multiply reg_weight, align guidance The scale of the gradient (including guidance_weight)
                reg_norms.append((reg_weight * self.reg_grads[i]).norm().item())

        if guid_norms:
            avg_guid = sum(guid_norms) / len(guid_norms)
            log["grad_norm/guidance"] = avg_guid
            if reg_norms:
                avg_reg = sum(reg_norms) / len(reg_norms)
                log["grad_norm/reg"] = avg_reg
                log["grad_norm/ratio"] = avg_guid / max(avg_reg, 1e-8)

        return log


# =====================================================================
# VelocityTracker: FlowEdit single step denoising velocity proxy logger
# =====================================================================

@dataclass
class VelocityTracker:
    """Per-step velocity proxy for FlowEdit denoise: separate leaf for guidance vs reg grads."""

    # P3 write
    v_student: Optional[torch.Tensor] = None     # (N,C), Yes autograd Picture arrived θ
    v_proxy: Optional[torch.Tensor] = None       # (N,C), detach leaf

    # P3.5 write(reg_weight > 0 time)
    reg_grad: Optional[torch.Tensor] = None      # (N,C), detached
    reg_loss_val: Optional[float] = None          # scalar

    def setup_proxy(self, v_student: torch.Tensor):
        """P3:Save v_student, create proxy leaf."""
        self.v_student = v_student
        self.v_proxy = v_student.detach().requires_grad_(True)

    def relay_and_backward(self):
        """P5: merge guidance + reg gradient, once backward Arrive θ."""
        total_grad = self.v_proxy.grad
        if self.reg_grad is not None:
            total_grad = total_grad + self.reg_grad
        self.v_student.backward(total_grad)

    def collect_log(self, reg_weight: float = 0.0) -> Dict[str, float]:
        """
        Collect gradients norm log,key with RolloutTracker.collect_log consistent.

        Args:
            reg_weight: reg weight (for log reference only,reg_grad weight included).

        Returns:
            Log dictionary, which may contain:
            - loss/reg:           reg loss scalar value
            - grad_norm/guidance: guidance gradient L2 norm (v_proxy level)
            - grad_norm/reg:     reg gradient L2 norm (v_proxy level)
            - grad_norm/ratio:   guidance / reg gradient norm ratio
        """
        log: Dict[str, float] = {}

        # loss/reg
        if self.reg_loss_val is not None:
            log["loss/reg"] = self.reg_loss_val

        # grad_norm/guidance — v_proxy.grad(P4c backward padding)
        guid_norm = 0.0
        if self.v_proxy is not None and self.v_proxy.grad is not None:
            guid_norm = self.v_proxy.grad.norm().item()
            log["grad_norm/guidance"] = guid_norm

        # grad_norm/reg — reg_grad(P3.5 backward padding)
        reg_norm = 0.0
        if self.reg_grad is not None:
            reg_norm = self.reg_grad.norm().item()
            log["grad_norm/reg"] = reg_norm

        # grad_norm/ratio
        if guid_norm > 0 and reg_norm > 0:
            log["grad_norm/ratio"] = guid_norm / max(reg_norm, 1e-8)

        return log
