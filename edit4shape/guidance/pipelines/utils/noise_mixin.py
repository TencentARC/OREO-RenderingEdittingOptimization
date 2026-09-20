"""
噪声管理 Mixin。

包含 BaseNoiseMixin，支持普通 Qwen FlowEdit 所需的 random / fixed /
aligned / delta 模式。
"""

from typing import Optional, Literal
import torch


# =============================================================================
# 噪声模式类型
# =============================================================================

NoiseMode = Literal[
    "random",
    "fixed",
    "aligned",
    "delta",
]


# =============================================================================
# BaseNoiseMixin - 噪声管理基类
# =============================================================================

class BaseNoiseMixin:
    """
    噪声管理基类。
    
    支持的噪声模式：
    - random: 每步随机采样
    - fixed: 固定噪声（初始化后不变）
    - aligned: DNAEdit 风格累积补偿 ε -= (v_cond - v_uncond) * (1 - t)
    - delta: 双分支差分补偿 ε -= v_delta * dt，update() 时需传入 v_delta
    
    使用方法：
        @dataclass
        class MyTracker(BaseStateTracker, BaseNoiseMixin):
            _noise: Optional[torch.Tensor] = None
            _mode: NoiseMode = "fixed"
    """
    
    # 子类需要定义这些字段
    _noise: Optional[torch.Tensor]
    _mode: NoiseMode
    
    def init(
        self, 
        x_src: torch.Tensor, 
        mode: NoiseMode = "fixed",
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        初始化噪声。
        
        Args:
            x_src: [B, seq, C] 源 latent
            mode: 噪声模式
            seed: 随机种子
        
        Returns:
            [B, seq, C] 初始噪声
        """
        self._mode = mode
        
        if seed is not None:
            generator = torch.Generator(device=x_src.device).manual_seed(seed)
            self._noise = torch.randn(
                x_src.shape, generator=generator, device=x_src.device, dtype=x_src.dtype
            )  # [B, seq, C]
        else:
            self._noise = torch.randn_like(x_src)  # [B, seq, C]
        
        return self._noise
    
    def get_noise(self, x_src: torch.Tensor) -> torch.Tensor:
        """
        获取当前噪声。
        
        Args:
            x_src: [B, seq, C] 参考 tensor
        
        Returns:
            [B, seq, C] 噪声
        """
        if self._noise is None:
            raise RuntimeError("请先调用 init() 初始化噪声")
        
        if self._mode == "random":
            return torch.randn_like(x_src)  # [B, seq, C]
        return self._noise  # [B, seq, C]
    
    def update(
        self,
        v_cond: torch.Tensor,    # 条件速度
        v_uncond: torch.Tensor,  # 无条件速度
        v_cfg: torch.Tensor,     # CFG 速度
        t: float,                # 时间步 [0, 1]
        dt: float,               # 时间步差（必传）
        v_delta: Optional[torch.Tensor] = None,  # delta 模式必传：v_cfg_tgt - v_cfg_src
        **kwargs,                # 忽略额外调用参数
    ) -> None:
        """
        更新噪声。
        
        aligned 模式：ε -= (v_cond - v_uncond) * (1 - t)
        delta 模式：ε -= v_delta * dt，保证 z_t_tgt = (1-t)*z_edit + t*ε₀
        
        Args:
            v_cond: [B, seq, C] 条件速度
            v_uncond: [B, seq, C] 无条件速度
            v_cfg: [B, seq, C] CFG 速度
            t: 当前时间步 [0, 1]
            dt: 时间步差（t_prev - t_curr 或 t_curr - t_prev，取决于调用方向）
            v_delta: [B, seq, C] 差分速度（delta 模式必传）
            **kwargs: 忽略其他参数
        """
        if self._mode == "aligned":
            # DNAEdit 风格累积补偿
            v_delta = v_cond - v_uncond  # [B, seq, C]
            self._noise = self._noise.to(torch.float32)
            self._noise -= v_delta.to(torch.float32) * (1.0 - t)
            self._noise = self._noise.to(v_delta.dtype)
        
        elif self._mode == "delta":
            # 双分支差分补偿：ε -= v_delta * dt
            if v_delta is None:
                raise ValueError("delta 模式必须传入 v_delta（v_cfg_tgt - v_cfg_src）")
            self._noise = self._noise.to(torch.float32)  # [B, seq, C]
            self._noise -= v_delta.to(torch.float32) * dt  # [B, seq, C]
            self._noise = self._noise.to(v_delta.dtype)  # [B, seq, C]
    
    @property
    def noise(self) -> Optional[torch.Tensor]:
        """当前噪声 [B, seq, C]"""
        return self._noise
    
    @property
    def mode(self) -> NoiseMode:
        """当前噪声模式"""
        return self._mode
