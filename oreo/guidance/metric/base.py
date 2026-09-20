"""Metric Base class definition."""
from abc import ABC, abstractmethod
from typing import Optional, Any

import torch


class BaseMetric(ABC):
    """
    Similarity Metric Base class.

    all metric Must be achieved compute Method, returns a scalar loss.
    """

    name: str = "base"  # Subclasses must override

    def __init__(self, weight: float, device: torch.device, **kwargs):
        """
        initialization Metric.

        Args:
            weight: loss Weights (for logging only, actual weighting is done externally)
            device: computing equipment
            **kwargs: subclass specific parameters
        """
        self.weight = weight
        self.device = device

    @abstractmethod
    def compute(
        self,
        rendered: torch.Tensor,
        target: Any,
        **kwargs,
    ) -> Optional[torch.Tensor]:
        """
        Calculate loss(without weighting).

        Args:
            rendered: Rendering (B,C,H,W) [0,1], with gradient
            target: target (can be an image tensor or other format)
            **kwargs: extra parameters

        Returns:
            scalar loss, or None(if not calculated)
        """
        pass

    def cleanup(self) -> None:
        """Release resources (optional implementation)."""
        pass








