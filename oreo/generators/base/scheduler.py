"""
Flow Matching scheduler base — minimal interface shared by Trellis schedulers.

Exposes timesteps / sigmas used by sde_step_with_logprob and ODE rollouts.
"""
import torch
from typing import Optional


class FlowScheduler:
    """
    Flow Matching scheduler base, compatible with sde_step_with_logprob.
    
    Properties:
        timesteps: discrete time indices
        sigmas: noise levels along the path
    """

    def __init__(self):
        self.timesteps: torch.Tensor = torch.tensor([])

    @property
    def sigmas(self) -> torch.Tensor:
        """Flow Matching in sigma = t"""
        return self.timesteps

    def set_timesteps(self, num_steps: int, device: torch.device, rescale_t: float = 1.0):
        """
        Set the time step sequence (from1Arrive0, a total of num_steps+1 points).

        Args:
            num_steps: Number of sampling steps
            device: Equipment
            rescale_t: Time step scaling factor (default1.0, no scaling)
        """
        t = torch.linspace(1.0, 0.0, num_steps + 1, device=device)  # (num_steps+1,)
        self.timesteps = rescale_t * t / (1 + (rescale_t - 1) * t)  # (num_steps+1,)

    def index_for_timestep(self, t: float) -> int:
        """Find time step index"""
        diffs = (self.timesteps - t).abs()  # (num_steps+1,)
        return int(diffs.argmin().item())
