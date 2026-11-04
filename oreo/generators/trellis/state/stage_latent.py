"""
StageLatent — per-stage latent bundle shared by TrellisState and TrellisContrastiveState.

Stage 1 (structure): z0 dense (B, 8, 16, 16, 16)
Stage 2 (SLAT): z0 SparseTensor feats/coords
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class StageLatent:
    """Single-stage latent container (z0 / noisy / velocity fields as used by each stage)."""
    z0: Any = None
    zt: Any = None
    z0_pred: Any = None
    z0_teacher: Any = None
    coords: Any = None       # Stage 1 only: argwhere(decoder(z₀) > 0)
    reg_loss: Any = None     # rollout regularization of output loss
