"""Noise-mode mixin for FlowEdit: random / fixed / aligned / delta."""

from typing import Optional, Literal
import torch


# =============================================================================
# Noise pattern type
# =============================================================================

NoiseMode = Literal[
    "random",
    "fixed",
    "aligned",
    "delta",
]


# =============================================================================
# BaseNoiseMixin - Noise management base class
# =============================================================================

class BaseNoiseMixin:
    """Controls how edit noise is sampled or reused across steps."""

    # Subclasses need to define these fields
    _noise: Optional[torch.Tensor]
    _mode: NoiseMode

    def init(
        self,
        x_src: torch.Tensor,
        mode: NoiseMode = "fixed",
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Initialize noise.

        Args:
            x_src: [B, seq, C] source latent
            mode: noise pattern
            seed: random seed

        Returns:
            [B, seq, C] initial noise
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
        Get the current noise.

        Args:
            x_src: [B, seq, C] Reference tensor

        Returns:
            [B, seq, C] noise
        """
        if self._noise is None:
            raise RuntimeError("Please call first init() initialization noise")

        if self._mode == "random":
            return torch.randn_like(x_src)  # [B, seq, C]
        return self._noise  # [B, seq, C]

    def update(
        self,
        v_cond: torch.Tensor,    # Conditional speed
        v_uncond: torch.Tensor,  # unconditional speed
        v_cfg: torch.Tensor,     # CFG speed
        t: float,                # time step [0, 1]
        dt: float,               # Time step (required)
        v_delta: Optional[torch.Tensor] = None,  # delta The pattern must be passed:v_cfg_tgt - v_cfg_src
        **kwargs,                # Ignore extra call arguments
    ) -> None:
        """
        Update noise.

        aligned Mode:ε -= (v_cond - v_uncond) * (1 - t)
        delta Mode:ε -= v_delta * dt, guaranteed z_t_tgt = (1-t)*z_edit + t*ε₀

        Args:
            v_cond: [B, seq, C] Conditional speed
            v_uncond: [B, seq, C] unconditional speed
            v_cfg: [B, seq, C] CFG speed
            t: current time step [0, 1]
            dt: time step(t_prev - t_curr or t_curr - t_prev, depending on the calling direction)
            v_delta: [B, seq, C] Differential speed (delta Pattern must be passed)
            **kwargs: Ignore other parameters
        """
        if self._mode == "aligned":
            # DNAEdit style cumulative compensation
            v_delta = v_cond - v_uncond  # [B, seq, C]
            self._noise = self._noise.to(torch.float32)
            self._noise -= v_delta.to(torch.float32) * (1.0 - t)
            self._noise = self._noise.to(v_delta.dtype)

        elif self._mode == "delta":
            # Dual branch differential compensation:ε -= v_delta * dt
            if v_delta is None:
                raise ValueError("delta Pattern must be passed in v_delta(v_cfg_tgt - v_cfg_src)")
            self._noise = self._noise.to(torch.float32)  # [B, seq, C]
            self._noise -= v_delta.to(torch.float32) * dt  # [B, seq, C]
            self._noise = self._noise.to(v_delta.dtype)  # [B, seq, C]

    @property
    def noise(self) -> Optional[torch.Tensor]:
        """current noise [B, seq, C]"""
        return self._noise

    @property
    def mode(self) -> NoiseMode:
        """Current noise pattern"""
        return self._mode
