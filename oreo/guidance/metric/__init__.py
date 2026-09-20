"""
Metric module.

Provides a unified similarity calculation interface and supports:
- MSE: Pixel space mean square error
- SSIM: structural similarity
- LPIPS: perceived similarity (VGG Features)
- DINO: DINOv3 feature similarity
- CLIP: CLIP Image feature similarity

Note:latent_csd and latent_mse by StateTracker.loss() Compute, don't register here.

How to use:
    from oreo.guidance.metric import create_metrics

    metrics = create_metrics(loss_cfg, device, extra_kwargs={...})
    for name, metric in metrics.items():
        loss = metric.compute(rendered, target)
"""
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
    """
    According to loss_cfg weight creation in metrics(weight > 0 only created).

    Args:
        loss_cfg: Configuration objects, including various metric weight (such as loss_cfg.ssim, loss_cfg.lpips)
        device: computing equipment
        extra_kwargs: Additional parameters, format {"metric_name": {"param": value}}

    Returns:
        dict: {name: metric} Contains only weight > 0 of metrics

    Example:
        >>> metrics = create_metrics(
        ...     loss_cfg,  # ssim=0.0, lpips=1.0, dino=0.5
        ...     device,
        ...     extra_kwargs={
        ...         "dino": {"model_path": "...", "image_size": 518},
        ...     }
        ... )
        >>> # Return {"lpips": LPIPSMetric(...), "dino": DINOMetric(...)}
    """
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








