"""Visualization mixin: decode uniform step samples and build progress grids."""

from typing import List, Any
import torch
from PIL import Image


class VisualizationMixin:
    """Progress visualization helpers for trackers."""

    # Subclasses need to define these fields
    x0_preds: List[torch.Tensor]
    height: int
    width: int
    images: List[Image.Image]

    @property
    def step_latents(self) -> List[torch.Tensor]:
        """Return to be visualized latent list, default is x0_preds"""
        return self.x0_preds

    @property
    def has_images(self) -> bool:
        """Have you decode image"""
        return len(self.images) > 0

    def decode_uniform_samples(self, pipe: Any, n_samples: int = 4) -> None:
        """
        uniform sampling n_samples Parallelize intermediate steps decode.

        Args:
            pipe: Pipeline(Required _decode_latent_to_image method)
            n_samples: The number of sampling steps must be a perfect square number (4, 9, 16, ...)

        The results are stored in self.images.
        """
        K = len(self.step_latents)
        if K == 0:
            return

        # Calculate the index of uniform sampling
        if n_samples >= K:
            indices = list(range(K))
            n_samples = K
        else:
            indices = [int(i * (K - 1) / (n_samples - 1)) for i in range(n_samples)]

        # collection decode of latents
        sampled_latents = [self.step_latents[i] for i in indices]

        # Fixed to single frame decoding to avoid OOM
        max_batch = 1

        # in batches decode
        decoded_images = []
        with torch.no_grad():
            for start in range(0, n_samples, max_batch):
                chunk_latents = torch.cat(
                    sampled_latents[start : start + max_batch],
                    dim=0
                )  # [batch, seq_len, C]
                chunk_images = pipe._decode_latent_to_image(
                    chunk_latents,
                    self.height,
                    self.width,
                    "pil"
                )
                decoded_images.extend(chunk_images)

        self.images = decoded_images

    def get_progress_grid(self, pipe: Any, n_samples: int = 4) -> Image.Image:
        """
        will n_samples Combine the intermediate images into one √n × √n Grid diagram.

        Args:
            pipe: Pipeline
            n_samples: The number of sampling steps must be a perfect square number (4, 9, 16, ...)

        Returns:
            synthetic mesh PIL Image
        """
        # Check if it is a perfect square number
        grid_size = int(n_samples ** 0.5)
        assert grid_size * grid_size == n_samples, f"n_samples must be a perfect square, got {n_samples}"

        # Decode Intermediate step (if not already decode or the quantity is wrong)
        if not self.has_images or len(self.images) != n_samples:
            self.decode_uniform_samples(pipe, n_samples)

        # synthetic mesh
        img_w, img_h = self.images[0].size
        grid_img = Image.new("RGB", (img_w * grid_size, img_h * grid_size), (255, 255, 255))

        for i, img in enumerate(self.images):
            row = i // grid_size
            col = i % grid_size
            grid_img.paste(img, (col * img_w, row * img_h))

        return grid_img
