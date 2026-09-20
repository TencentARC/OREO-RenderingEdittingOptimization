"""Qwen-Image FlowEdit state tracker.

The retained FlowEdit pipeline records source/target x0 predictions with
StateTracker and the random/fixed/aligned/delta BaseNoiseMixin modes.
"""

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
# StateTracker - 统一状态追踪器
# =============================================================================

@dataclass
class StateTracker(BaseStateTracker, LossMixin, VisualizationMixin, BaseNoiseMixin):
    """
    统一状态追踪器（random / fixed / aligned / delta 模式）。
    
    存储：
    - x0_preds: 预测的 x0（MSE loss 目标）
    - x0_pos: CSD 正样本（高 CFG x0 预测，吸引）
    - x0_neg: CSD 负样本（低 CFG x0 预测，排斥）
    - ts: 时间步列表
    
    通过 mse_weight / csd_weight 控制 loss 类型：
    - mse_weight=1 → 纯 MSE: MSE(src, x0_pred)
    - csd_weight=1 → 纯 CSD: MSE(src, x0_pos) - MSE(src, x0_neg)
    
    噪声模式：
    - random / fixed / aligned: 见 BaseNoiseMixin
    - delta: 双分支差分补偿 ε -= v_delta * dt，update() 时需传入 v_delta 和 dt
    
    使用方式：
        tracker = StateTracker(height=H, width=W)
        tracker.init(x_src, mode="aligned", seed=0)
        for t in timesteps:
            noise = tracker.get_noise(x_src)
            z_t = (1 - t) * x_src + t * noise
            v_cond, v_uncond, v_cfg = model(z_t, t)
            x0_pred = z_t - t * v_cfg
            x0_pos = z_t - t * v_cond
            x0_neg = z_t - t * v_uncond
            tracker.record(x0_pred, t, x0_pos, x0_neg)
            tracker.update(v_cond, v_uncond, v_cfg, t, dt)  # delta 模式需传 v_delta=...
    """
    
    # 预测结果
    x0_preds: List[torch.Tensor] = field(default_factory=list)    # MSE 目标
    x0_pos: List[torch.Tensor] = field(default_factory=list)      # CSD 正样本（吸引）
    x0_neg: List[torch.Tensor] = field(default_factory=list)      # CSD 负样本（排斥）
    ts: List[float] = field(default_factory=list)                  # 时间步
    
    # 尺寸
    height: int = None
    width: int = None
    
    # 可视化
    image: Image.Image = None
    images: List[Image.Image] = field(default_factory=list)
    
    # 噪声管理（BaseNoiseMixin）
    _noise: Optional[torch.Tensor] = None
    _mode: NoiseMode = "fixed"
    
    # =========================================================================
    # 记录方法
    # =========================================================================
    
    def record(
        self, 
        x0_pred: torch.Tensor,
        t: float,
        x0_pos: torch.Tensor,
        x0_neg: torch.Tensor,
    ) -> None:
        """
        记录一步状态。
        
        Args:
            x0_pred: [B, seq, C] 预测的 x0（MSE 目标）
            t: 当前时间步
            x0_pos: [B, seq, C] CSD 正样本（高 CFG x0 预测，吸引）
            x0_neg: [B, seq, C] CSD 负样本（低 CFG x0 预测，排斥）
        """
        self.x0_preds.append(x0_pred.detach().clone())  # [B, seq, C]
        self.x0_pos.append(x0_pos.detach().clone())  # [B, seq, C]
        self.x0_neg.append(x0_neg.detach().clone())  # [B, seq, C]
        self.ts.append(t)
    
    # =========================================================================
    # 属性
    # =========================================================================
    
    @property
    def target(self) -> torch.Tensor:
        """目标 latent [B, seq, C]，返回最后一个 x0_pred"""
        if not self.x0_preds:
            return None
        return self.x0_preds[-1]  # [B, seq, C]
    
    @property
    def num_steps(self) -> int:
        """返回记录的步数"""
        return len(self.x0_preds)
    
    @property
    def final(self) -> torch.Tensor:
        """target 的别名"""
        return self.target
    
    def stack(self) -> torch.Tensor:
        """堆叠所有 x0_pred [K, B, seq, C]"""
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
    """Create the retained FlowEdit tracker for a supported noise mode."""
    if noise_mode not in {"random", "fixed", "aligned", "delta"}:
        raise ValueError(
            f"Unsupported FlowEdit noise mode: {noise_mode!r}; "
            "expected random, fixed, aligned, or delta"
        )
    return StateTracker(height=height, width=width)
