"""BaseStateTracker shared by FlowEdit / CSD / SDS-style trackers."""

from dataclasses import dataclass
from typing import Any, List, Optional
import torch
from PIL import Image


@dataclass
class BaseStateTracker:
    """Abstract tracker API all guidance methods inherit."""

    height: int = 0
    width: int = 0

    # =========================================================================
    # shared methods
    # =========================================================================

    def _decode_latent(self, pipe: Any, latent: torch.Tensor) -> Optional[Image.Image]:
        """
        Decode single latent for images.

        Args:
            pipe: Pipeline(Required _decode_latent_to_image method)
            latent: [1, seq, C]

        Returns:
            PIL Image
        """
        with torch.no_grad():
            images = pipe._decode_latent_to_image(
                latent,
                self.height,
                self.width,
                "pil"
            )
            return images[0] if images else None

    def _concat_images_horizontal(self, *images: Image.Image) -> Optional[Image.Image]:
        """
        Stitch multiple images horizontally.

        Args:
            *images: PIL Image list

        Returns:
            spliced PIL Image
        """
        images = [img for img in images if img is not None]
        if not images:
            return None

        total_width = sum(img.width for img in images)
        max_height = max(img.height for img in images)

        combined = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        for img in images:
            combined.paste(img, (x_offset, 0))
            x_offset += img.width

        return combined
