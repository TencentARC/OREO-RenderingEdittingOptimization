"""Pixel-space MSE between rendered and target images."""
import logging
from typing import Optional

import torch
import torch.nn.functional as F

from .base import BaseMetric


class MSEMetric(BaseMetric):
    """Mean squared error in image space."""

    name = "mse"

    def __init__(self, weight: float, device: torch.device, **kwargs):
        super().__init__(weight, device)
        logging.info(f"[MSEMetric] Initialized (weight={weight})")

    def compute(
        self,
        rendered: torch.Tensor,  # (B,C,H,W) [0,1], with gradient
        target: torch.Tensor,    # (B,C,H,W) [0,1], no gradient
        **kwargs,
    ) -> Optional[torch.Tensor]:
        """Compute pixel space MSE loss."""
        return F.mse_loss(rendered, target.detach())  # scalar
