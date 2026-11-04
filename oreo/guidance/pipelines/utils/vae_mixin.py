"""Differentiable VAE encode mixin for guidance pipelines (keeps grads into latents)."""

import torch
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import retrieve_latents


class DifferentiableVAEMixin:
    """Requires the host Pipeline to expose VAE encode hooks."""

    def _encode_vae_image_differentiable(self, image: torch.Tensor) -> torch.Tensor:
        """
        Differentiable version VAE encode(without @torch.no_grad).

        with _encode_vae_image Same logic, but retaining gradients for backpropagation.

        Args:
            image: [B, C, 1, H, W] images,[-1, 1] range,bfloat16

        Returns:
            normalized latent [B, C_lat, 1, H_lat, W_lat]
        """
        # VAE encode, retain the gradient
        image_latents = retrieve_latents(self.vae.encode(image), sample_mode="argmax")

        # Standardization
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.latent_channels, 1, 1, 1)
            .to(image_latents.device, image_latents.dtype)
        )  # [1, C_lat, 1, 1, 1]
        latents_std = (
            torch.tensor(self.vae.config.latents_std)
            .view(1, self.latent_channels, 1, 1, 1)
            .to(image_latents.device, image_latents.dtype)
        )  # [1, C_lat, 1, 1, 1]
        return (image_latents - latents_mean) / latents_std  # [B, C_lat, 1, H_lat, W_lat]
