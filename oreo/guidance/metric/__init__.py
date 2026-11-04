"""Similarity metrics used inside FlowEdit / contrastive losses (MSE, SSIM, LPIPS, CLIP, DINO)."""
from typing import Dict, Any

import torch

from .base import BaseMetric
from .mse import MSEMetric
from .ssim import SSIMMetric
from .lpips import LPIPSMetric
from .dino import DINOMetric
from .clip import CLIPMetric


# Registry:name -> class
# Note:latent_csd/latent_mse Do not register here, by Tracker.loss() Process
METRIC_REGISTRY: Dict[str, type] = {
    "mse": MSEMetric,
    "ssim": SSIMMetric,
    "lpips": LPIPSMetric,
    "dino": DINOMetric,
    "clip": CLIPMetric,
}


def create_metrics(
    loss_cfg: Any,
    device: torch.device,
    extra_kwargs: Dict[str, Dict] = None,
) -> Dict[str, BaseMetric]:
    """Instantiate metrics whose configured weight > 0."""
    extra_kwargs = extra_kwargs or {}
    metrics = {}

    for name, cls in METRIC_REGISTRY.items():
        # Get weight: must be in config Explicit declaration in
        weight = loss_cfg[name]

        if weight > 0:
            kwargs = extra_kwargs.get(name, {})
            metrics[name] = cls(weight=weight, device=device, **kwargs)

    return metrics


__all__ = [
    "BaseMetric",
    "MSEMetric",
    "SSIMMetric",
    "LPIPSMetric",
    "DINOMetric",
    "CLIPMetric",
    "METRIC_REGISTRY",
    "create_metrics",
]







