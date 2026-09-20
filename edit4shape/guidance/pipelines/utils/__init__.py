"""
Pipeline 工具模块。

包含:
- 基类: BaseStateTracker
- 噪声管理: BaseNoiseMixin, NoiseMode
- Loss 函数: mse_loss_step, contrastive_loss_step, mse_loss, csd_loss
- 可视化: VisualizationMixin
- VAE Mixin: DifferentiableVAEMixin
"""

from .base import BaseStateTracker
from .noise_mixin import BaseNoiseMixin, NoiseMode
from .loss_functions import (
    # 类型
    ReduceMode,
    # 工具函数
    reduce_losses,
    normalize_grad,
    # 单步 Loss
    mse_loss_step,
    contrastive_loss_step,
    # 多步 Loss
    mse_loss,
    csd_loss,
    # Mixin
    LossMixin,
)
from .visualization import VisualizationMixin
from .vae_mixin import DifferentiableVAEMixin


__all__ = [
    # 基类
    "BaseStateTracker",
    # 噪声管理
    "BaseNoiseMixin",
    "NoiseMode",
    # Loss 类型
    "ReduceMode",
    # Loss 工具
    "reduce_losses",
    "normalize_grad",
    # 单步 Loss
    "mse_loss_step",
    "contrastive_loss_step",
    # 多步 Loss
    "mse_loss",
    "csd_loss",
    # Loss Mixin
    "LossMixin",
    # 可视化
    "VisualizationMixin",
    # VAE Mixin
    "DifferentiableVAEMixin",
]
