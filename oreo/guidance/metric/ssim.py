"""SSIM loss."""
import logging
from typing import Optional

import torch
from pytorch_msssim import ssim

from .base import BaseMetric


class SSIMMetric(BaseMetric):
    """
    SSIM Similarity Metric.

    SSIM The higher they are, the more similar they are,loss = 1 - SSIM.
    """

    name = "ssim"

    def __init__(self, weight: float, device: torch.device, **kwargs):
        super().__init__(weight, device)
        logging.info(f"[SSIMMetric] Initialized (weight={weight})")

    def compute(
        self,
        rendered: torch.Tensor,  # (B,C,H,W) [0,1], with gradient
        target: torch.Tensor,    # (B,C,H,W) [0,1], no gradient
        **kwargs,
    ) -> Optional[torch.Tensor]:
        """Calculate SSIM loss."""
        ssim_val = ssim(rendered, target, data_range=1.0, size_average=True)  # scalar
        return 1 - ssim_val








