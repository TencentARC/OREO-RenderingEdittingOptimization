"""CLIP Image feature similarity loss."""
import logging
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

from .base import BaseMetric


class CLIPMetric(BaseMetric):
    """
    CLIP Image feature space similarity Metric.

    Use CLIP Vision Encoder Extract image-level features and calculate cosine similarity,loss = 1 - similarity.
    Stay derivable:rendered Keep the gradient,target No gradient.

    Use independent CLIP Model implementation retained FlowEdit metric.
    """

    name = "clip"

    def __init__(
        self,
        weight: float,
        device: torch.device,
        model_path: str = "pretrained_weights/clip/clip-vit-large-patch14",
        image_size: int = 224,
        **kwargs,
    ):
        super().__init__(weight, device)

        logging.info(f"[CLIPMetric] Loading CLIP: {model_path}")
        self.processor = CLIPProcessor.from_pretrained(model_path)
        self.model = CLIPModel.from_pretrained(model_path, torch_dtype=torch.float32).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad = False

        # CLIP Normalization parameters (OpenAI CLIP standard value)
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)  # (1,3,1,1)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)   # (1,3,1,1)
        self.size = image_size
        logging.info(f"[CLIPMetric] Initialized (weight={weight}, size={image_size})")

    def compute(
        self,
        rendered: torch.Tensor,  # (B,C,H,W) [0,1], with gradient
        target: torch.Tensor,    # (B,C,H,W) [0,1], no gradient
        **kwargs,
    ) -> Optional[torch.Tensor]:
        """Calculate CLIP Image feature similarity loss."""
        # Resize(derivable)
        r = F.interpolate(rendered, size=(self.size, self.size), mode='bilinear', align_corners=False)  # (B,3,size,size)
        t = F.interpolate(target.detach(), size=(self.size, self.size), mode='bilinear', align_corners=False)  # (B,3,size,size)

        # CLIP Normalization (differentiable)
        r = (r - self.mean) / self.std  # (B,3,size,size)
        t = (t - self.mean) / self.std  # (B,3,size,size)

        # Extract image-level features (rendered Keep the gradient,target no gradient)
        feats_r = self.model.get_image_features(pixel_values=r)  # (B, D)
        with torch.no_grad():
            feats_t = self.model.get_image_features(pixel_values=t)  # (B, D)

        # cosine similarity
        feats_r = F.normalize(feats_r, dim=-1)  # (B, D)
        feats_t = F.normalize(feats_t.detach(), dim=-1)  # (B, D)
        sim = (feats_r * feats_t).sum(dim=-1).mean()  # scalar

        return 1 - sim  # loss

    @torch.no_grad()
    def compute_from_pil(
        self,
        rendered_pils: list[Image.Image],
        target_pils: list[Image.Image],
        **kwargs,
    ) -> Optional[torch.Tensor]:
        """PIL Enter the path, use CLIPProcessor preprocessing."""
        r_inputs = self.processor(images=rendered_pils, return_tensors="pt", padding=True)
        t_inputs = self.processor(images=target_pils, return_tensors="pt", padding=True)

        r = r_inputs["pixel_values"].to(self.device)  # (B,3,H,W)
        t = t_inputs["pixel_values"].to(self.device)  # (B,3,H,W)

        out_r = self.model.vision_model(pixel_values=r)  # (B,L,D)
        out_t = self.model.vision_model(pixel_values=t)  # (B,L,D)
        feats_r = out_r.last_hidden_state[:, 0]  # (B, D)
        feats_t = out_t.last_hidden_state[:, 0]  # (B, D)

        feats_r = F.normalize(feats_r, dim=-1)  # (B, D)
        feats_t = F.normalize(feats_t, dim=-1)  # (B, D)
        sim = (feats_r * feats_t).sum(dim=-1).mean()  # scalar

        return 1 - sim  # loss

    def cleanup(self) -> None:
        if hasattr(self, 'model'):
            del self.model
            logging.info("[CLIPMetric] Cleaned up.")
