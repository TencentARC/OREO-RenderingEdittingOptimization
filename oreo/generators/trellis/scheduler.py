"""Trellis Flow Matching Scheduler"""
import torch
from types import SimpleNamespace
from typing import Optional, Union

from trellis.modules.sparse import SparseTensor

from oreo.generators.base.scheduler import FlowScheduler
from oreo.generators.base.rollout import sde_step_with_logprob_sparse


class TrellisFlowScheduler(FlowScheduler):
    """
    Trellis dedicated Flow Matching scheduler.

    Inherited from base class set_timesteps,sigmas,index_for_timestep,
    additionally provided SparseTensor of ODE step and SDE step method.
    """

    def step(self, noise_pred: SparseTensor, t: Union[float, torch.Tensor], latents: SparseTensor) -> SimpleNamespace:
        """
        Euler ODE Step:x_{t-1} = x_t - (t - t_prev) * v

        Args:
            noise_pred: Velocity field prediction v,SparseTensor
            t: current time step, scalar
            latents: current sample x_t,SparseTensor

        Returns:
            SimpleNamespace(prev_sample=SparseTensor, pred_original_sample=None)
        """
        t_val = float(t)
        idx = self.index_for_timestep(t_val)
        t_prev = float(self.timesteps[idx + 1].item())
        delta = t_val - t_prev  # time difference

        pred_feats = latents.feats - delta * noise_pred.feats  # (N, C)
        prev_sample = SparseTensor(coords=latents.coords, feats=pred_feats)

        return SimpleNamespace(prev_sample=prev_sample, pred_original_sample=None)

    def sde_step(
        self,
        noise_pred: SparseTensor,
        t: Union[float, torch.Tensor],
        latents: SparseTensor,
        noise_level: float = 0.7,
        prev_sample: Optional[SparseTensor] = None,
        generator: Optional[torch.Generator] = None,
        sde_type: str = 'sde',
        return_sqrt_dt: bool = False,
    ):
        """
        SDE step, call rollout.sde_step_with_logprob_sparse.

        Args:
            noise_pred: Velocity field prediction v,SparseTensor
            t: current time step, scalar float
            latents: current sample x_t,SparseTensor
            noise_level: Noise intensity (Default 0.7)
            prev_sample: If provided, use this sample calculation log_prob
            generator: random number generator
            sde_type: 'sde' or 'cps'
            return_sqrt_dt: Whether to return sqrt_dt

        Returns:
            prev_sample: SparseTensor
            log_prob: Tensor (B,)
            prev_sample_mean: SparseTensor
            std_dev_t: Tensor
            (Optional) sqrt_dt: Tensor
        """
        return sde_step_with_logprob_sparse(
            self,
            model_output=noise_pred,
            timestep=float(t),
            sample=latents,
            noise_level=noise_level,
            prev_sample=prev_sample,
            generator=generator,
            sde_type=sde_type,
            return_sqrt_dt=return_sqrt_dt,
        )
