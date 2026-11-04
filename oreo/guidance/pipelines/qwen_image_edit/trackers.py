"""Qwen-Image FlowEdit StateTracker: records src/tgt x0 preds under random/fixed/aligned/delta noise."""

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
    """Stores x0_preds and related tensors for multi-step MSE / contrastive losses."""

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
    """Build the retained FlowEdit tracker for a supported noise mode."""
    if noise_mode not in {"random", "fixed", "aligned", "delta"}:
        raise ValueError(
            f"Unsupported FlowEdit noise mode: {noise_mode!r}; "
            "expected random, fixed, aligned, or delta"
        )
    return StateTracker(height=height, width=width)
