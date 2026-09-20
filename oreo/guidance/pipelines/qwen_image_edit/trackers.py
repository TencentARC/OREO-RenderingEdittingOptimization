"""Qwen-Image FlowEdit state tracker.

The retained FlowEdit pipeline records source/target x0 predictions with
StateTracker and the random/fixed/aligned/delta BaseNoiseMixin modes.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import torch
from PIL import Image

from ..utils import (
    BaseStateTracker,
    BaseNoiseMixin,
    NoiseMode,
    VisualizationMixin,
    LossMixin,
)


# =============================================================================
# StateTracker - Unified status tracker
# =============================================================================

@dataclass
class StateTracker(BaseStateTracker, LossMixin, VisualizationMixin, BaseNoiseMixin):
    """
    Unified Status Tracker (random / fixed / aligned / delta mode).

    Storage:
    - x0_preds: Predictive x0(MSE loss target)
    - x0_pos: CSD Positive sample (high CFG x0 predict, attract)
    - x0_neg: CSD Negative samples (low CFG x0 prediction, rejection)
    - ts: time step list

    Pass mse_weight / csd_weight control loss Type:
    - mse_weight=1 → pure MSE: MSE(src, x0_pred)
    - csd_weight=1 → pure CSD: MSE(src, x0_pos) - MSE(src, x0_neg)

    Noise pattern:
    - random / fixed / aligned: see BaseNoiseMixin
    - delta: Dual branch differential compensation ε -= v_delta * dt,update() Need to be passed in v_delta and dt

    How to use:
        tracker = StateTracker(height=H, width=W)
        tracker.init(x_src, mode="aligned", seed=0)
        for t in timesteps:
            noise = tracker.get_noise(x_src)
            z_t = (1 - t) * x_src + t * noise
            v_cond, v_uncond, v_cfg = model(z_t, t)
            x0_pred = z_t - t * v_cfg
            x0_pos = z_t - t * v_cond
            x0_neg = z_t - t * v_uncond
            tracker.record(x0_pred, t, x0_pos, x0_neg)
            tracker.update(v_cond, v_uncond, v_cfg, t, dt)  # delta The mode needs to be passed v_delta=...
    """

    # Prediction results
    x0_preds: List[torch.Tensor] = field(default_factory=list)    # MSE target
    x0_pos: List[torch.Tensor] = field(default_factory=list)      # CSD Positive sample (attract)
    x0_neg: List[torch.Tensor] = field(default_factory=list)      # CSD Negative samples (exclusion)
    ts: List[float] = field(default_factory=list)                  # time step

    # Size
    height: int = None
    width: int = None

    # Visualization
    image: Image.Image = None
    images: List[Image.Image] = field(default_factory=list)

    # Noise Management (BaseNoiseMixin)
    _noise: Optional[torch.Tensor] = None
    _mode: NoiseMode = "fixed"

    # =========================================================================
    # Recording method
    # =========================================================================

    def record(
        self,
        x0_pred: torch.Tensor,
        t: float,
        x0_pos: torch.Tensor,
        x0_neg: torch.Tensor,
    ) -> None:
        """
        Record the status of a step.

        Args:
            x0_pred: [B, seq, C] Predictive x0(MSE target)
            t: current time step
            x0_pos: [B, seq, C] CSD Positive sample (high CFG x0 predict, attract)
            x0_neg: [B, seq, C] CSD Negative samples (low CFG x0 prediction, rejection)
        """
        self.x0_preds.append(x0_pred.detach().clone())  # [B, seq, C]
        self.x0_pos.append(x0_pos.detach().clone())  # [B, seq, C]
        self.x0_neg.append(x0_neg.detach().clone())  # [B, seq, C]
        self.ts.append(t)

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def target(self) -> torch.Tensor:
        """target latent [B, seq, C], return the last x0_pred"""
        if not self.x0_preds:
            return None
        return self.x0_preds[-1]  # [B, seq, C]

    @property
    def num_steps(self) -> int:
        """Returns the number of recorded steps"""
        return len(self.x0_preds)

    @property
    def final(self) -> torch.Tensor:
        """target alias"""
        return self.target

    def stack(self) -> torch.Tensor:
        """stack all x0_pred [K, B, seq, C]"""
        return torch.stack(self.x0_preds, dim=0)  # [K, B, seq, C]

    def __len__(self) -> int:
        return len(self.x0_preds)


# =============================================================================
# FlowEdit tracker factory
# =============================================================================

FlowEditTracker = StateTracker


def create_flowedit_tracker(
    noise_mode: str,
    height: int = None,
    width: int = None,
) -> FlowEditTracker:
    """Create the retained FlowEdit tracker for a supported noise mode."""
    if noise_mode not in {"random", "fixed", "aligned", "delta"}:
        raise ValueError(
            f"Unsupported FlowEdit noise mode: {noise_mode!r}; "
            "expected random, fixed, aligned, or delta"
        )
    return StateTracker(height=height, width=width)
