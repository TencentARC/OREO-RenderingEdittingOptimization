"""
Shared SDE / flow-matching step helpers used by Trellis rollouts.

Includes dense and SparseTensor variants of reverse SDE steps with optional log-prob.
"""

# Copied from https://github.com/kvablack/ddpo-pytorch/blob/main/ddpo_pytorch/diffusers_patch/ddim_with_logprob.py
# We adapt it from flow to flow matching.

import math
from typing import Optional, Union
import torch

from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

def sde_step_with_logprob(
    self: FlowMatchEulerDiscreteScheduler,
    model_output: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    sample: torch.FloatTensor,
    noise_level: float = 0.7,
    prev_sample: Optional[torch.FloatTensor] = None,
    generator: Optional[torch.Generator] = None,
    sde_type: Optional[str] = 'sde',
    return_sqrt_dt: Optional[bool] = False,
):
    """Reverse one SDE / flow step and optionally return log-probability of the transition."""
    # bf16 can overflow here when compute prev_sample_mean, we must convert all variable to fp32
    model_output=model_output.float()
    sample=sample.float()
    if prev_sample is not None:
        prev_sample=prev_sample.float()

    step_index = [self.index_for_timestep(t) for t in timestep]
    prev_step_index = [step+1 for step in step_index]
    sigma = self.sigmas[step_index].view(-1, *([1] * (len(sample.shape) - 1)))
    sigma_prev = self.sigmas[prev_step_index].view(-1, *([1] * (len(sample.shape) - 1)))
    sigma_max = self.sigmas[1].item()
    dt = sigma_prev - sigma

    if sde_type == 'sde':
        std_dev_t = torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max, sigma)))*noise_level

        # our sde
        prev_sample_mean = sample*(1+std_dev_t**2/(2*sigma)*dt)+model_output*(1+std_dev_t**2*(1-sigma)/(2*sigma))*dt

        if prev_sample is None:
            variance_noise = randn_tensor(
                model_output.shape,
                generator=generator,
                device=model_output.device,
                dtype=model_output.dtype,
            )
            prev_sample = prev_sample_mean + std_dev_t * torch.sqrt(-1*dt) * variance_noise

        log_prob = (
            -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * ((std_dev_t * torch.sqrt(-1*dt))**2))
            - torch.log(std_dev_t * torch.sqrt(-1*dt))
            - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
        )

    elif sde_type == 'cps':
        std_dev_t = sigma_prev  * math.sin(noise_level * math.pi / 2) # sigma_t in paper
        pred_original_sample = sample - sigma * model_output # predicted x_0 in paper
        noise_estimate = sample + model_output * (1 - sigma) # predicted x_1 in paper
        # detach x_1 (noise_estimate) Use a unified gradient direction (always negative) to avoid training instability caused by positive and negative offsets.
        prev_sample_mean = pred_original_sample * (1 - sigma_prev) + noise_estimate.detach() * torch.sqrt(sigma_prev**2 - std_dev_t**2)

        if prev_sample is None:
            variance_noise = randn_tensor(
                model_output.shape,
                generator=generator,
                device=model_output.device,
                dtype=model_output.dtype,
            )
            prev_sample = prev_sample_mean + std_dev_t * variance_noise

        # remove all constants
        log_prob = -((prev_sample.detach() - prev_sample_mean) ** 2)

    # mean along all but batch dimension
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    if return_sqrt_dt:
        return prev_sample, log_prob, prev_sample_mean, std_dev_t, torch.sqrt(-1*dt)
    return prev_sample, log_prob, prev_sample_mean, std_dev_t


def sde_step_with_logprob_sparse(
    self,  # FlowScheduler, need to provide sigmas and index_for_timestep
    model_output: "SparseTensor",
    timestep: float,
    sample: "SparseTensor",
    noise_level: float = 0.7,
    prev_sample: Optional["SparseTensor"] = None,
    generator: Optional[torch.Generator] = None,
    sde_type: str = 'sde',
    return_sqrt_dt: bool = False,
):
    """
    SparseTensor variant of the reverse SDE stepper for Trellis Flow Matching.
    
    Args:
        model_output: predicted velocity v as SparseTensor
    """
    # Use incoming sample type to avoid importing different modules SparseTensor cause isinstance Check failed
    SparseClass = sample.__class__

    # Extract feats, turn float32 Spill proof
    v = model_output.feats.float()  # (N, C)
    x_t = sample.feats.float()  # (N, C)
    x_prev = prev_sample.feats.float() if prev_sample is not None else None
    batch_indices = sample.coords[:, 0]  # (N,) batch Index
    orig_dtype = sample.feats.dtype

    # get sigma(Index Tensor still after Tensor)
    idx = self.index_for_timestep(timestep)
    sigma = self.sigmas[idx]  # Tensor scalar
    sigma_prev = self.sigmas[idx + 1]  # Tensor scalar
    sigma_max = self.sigmas[1]  # Tensor(Do not call .item())
    dt = sigma_prev - sigma  # negative value

    if sde_type == 'sde':
        # Use torch.where avoid Python Conditional expression type issues
        sigma_safe = torch.where(sigma == 1, sigma_max, sigma)
        std_dev_t = torch.sqrt(sigma_safe / (1 - sigma_safe)) * noise_level

        # SDE mean
        prev_sample_mean = x_t*(1+std_dev_t**2/(2*sigma)*dt)+v*(1+std_dev_t**2*(1-sigma)/(2*sigma))*dt  # (N, C)

        sqrt_neg_dt = torch.sqrt(-dt)
        if x_prev is None:
            variance_noise = randn_tensor(
                v.shape, generator=generator, device=v.device, dtype=v.dtype,
            )
            prev_sample_feats = prev_sample_mean + std_dev_t * sqrt_neg_dt * variance_noise  # (N, C)
        else:
            prev_sample_feats = x_prev

        scale = std_dev_t * sqrt_neg_dt
        log_prob = (
            -((prev_sample_feats.detach() - prev_sample_mean) ** 2) / (2 * scale**2)
            - torch.log(scale)
            - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
        )  # (N, C)

    elif sde_type == 'cps':
        std_dev_t = sigma_prev * math.sin(noise_level * math.pi / 2)
        pred_original_sample = x_t - sigma * v  # (N, C)
        noise_estimate = x_t + v * (1 - sigma)  # (N, C)
        # detach x_1 (noise_estimate) Use a unified gradient direction (always negative) to avoid training instability caused by positive and negative offsets.
        prev_sample_mean = pred_original_sample * (1 - sigma_prev) + noise_estimate.detach() * torch.sqrt(sigma_prev**2 - std_dev_t**2)  # (N, C)

        sqrt_neg_dt = torch.sqrt(-dt)
        if x_prev is None:
            variance_noise = randn_tensor(
                v.shape, generator=generator, device=v.device, dtype=v.dtype,
            )
            prev_sample_feats = prev_sample_mean + std_dev_t * variance_noise  # (N, C)
        else:
            prev_sample_feats = x_prev

        log_prob = -((prev_sample_feats.detach() - prev_sample_mean) ** 2)  # (N, C)

    else:
        raise ValueError(f"Unknown sde_type: {sde_type}")

    # press batch aggregation log_prob: (N, C) → (B,)
    log_prob = log_prob.mean(dim=-1)  # (N,)
    num_batches = int(batch_indices.max().item()) + 1
    log_prob_sum = torch.zeros(num_batches, device=log_prob.device, dtype=log_prob.dtype)  # (B,)
    counts = torch.zeros(num_batches, device=log_prob.device, dtype=log_prob.dtype)  # (B,)
    log_prob_sum.scatter_add_(0, batch_indices.long(), log_prob)  # (B,)
    counts.scatter_add_(0, batch_indices.long(), torch.ones_like(log_prob))  # (B,)
    log_prob = log_prob_sum / counts.clamp(min=1)  # (B,)

    # encapsulate back SparseTensor(Use dynamically obtained classes to avoid cross-module type mismatch)
    prev_sample_out = SparseClass(coords=sample.coords, feats=prev_sample_feats.to(orig_dtype))
    prev_sample_mean_out = SparseClass(coords=sample.coords, feats=prev_sample_mean.to(orig_dtype))

    if return_sqrt_dt:
        return prev_sample_out, log_prob, prev_sample_mean_out, std_dev_t, sqrt_neg_dt
    return prev_sample_out, log_prob, prev_sample_mean_out, std_dev_t