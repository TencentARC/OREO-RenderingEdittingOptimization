"""LPIPS perceptual loss (VGG-based feature distance)."""
import logging
from typing import Optional

import torch
import lpips as lpips_lib

from .base import BaseMetric


class LPIPSMetric(BaseMetric):
    """Lower LPIPS => more similar; used as a pixel-space perceptual term."""

    name = "lpips"

    def __init__(self, weight: float, device: torch.device, **kwargs):
        super().__init__(weight, device)

        logging.info(f"[LPIPSMetric] Loading VGG model...")
        self.fn = lpips_lib.LPIPS(net='vgg').to(device).eval()
        for p in self.fn.parameters():
            p.requires_grad = False
        logging.info(f"[LPIPSMetric] Initialized (weight={weight})")

    def compute(
        self,
        rendered: torch.Tensor,  # (B,C,H,W) [0,1], with gradient
        target: torch.Tensor,    # (B,C,H,W) [0,1], no gradient
        **kwargs,
    ) -> Optional[torch.Tensor]:
        """Calculate LPIPS loss."""
        # LPIPS need [-1, 1] range
        r = rendered * 2 - 1  # (B,C,H,W), [0,1] → [-1,1]
        t = target * 2 - 1    # (B,C,H,W), [0,1] → [-1,1]
        return self.fn(r, t).mean()  # scalar

    def cleanup(self) -> None:
        if hasattr(self, 'fn'):
            del self.fn
            logging.info("[LPIPSMetric] Cleaned up.")








