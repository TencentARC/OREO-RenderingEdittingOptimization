"""
StageLatent — single stage latent state container

Unified management of all flow matching stages latent status.
two State class (TrellisState, TrellisContrastiveState) share this definition.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class StageLatent:
    """single stage latent State container.

    Stage 1 (structure): z0 for dense Tensor (B, 8, 16, 16, 16)
    Stage 2 (SLAT):      z0 for SparseTensor (N, C)

    Attributes:
        z0:          clean latent — rollout Output (after denormalization)
        zt:          Noise latent — student training input
        z0_pred:     student Predictive ẑ₀
        z0_teacher:  contrastive teacher Reference(z₀ copy)
        coords:      Stage 1 unique — decoder(z₀) The sparse coordinates extracted after (N, 4)
        reg_loss:    rollout Output v-reg loss
    """
    z0: Any = None
    zt: Any = None
    z0_pred: Any = None
    z0_teacher: Any = None
    coords: Any = None       # Stage 1 only: argwhere(decoder(z₀) > 0)
    reg_loss: Any = None     # rollout regularization of output loss
