"""
Guidance Tool functions.

Provides general tools such as image format conversion.
"""

import base64
from io import BytesIO

import numpy as np
import torch
from PIL import Image


def tensor_to_base64(tensor: torch.Tensor) -> str:
    """
    will (C,H,W) Tensor Convert to Base64 string (PNG Format).

    Args:
        tensor: image tensor (C,H,W),float32 [0,1]

    Returns:
        str: Base64 coded PNG image
    """
    img_np = (tensor.detach().cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)  # (H,W,C)
    if img_np.shape[-1] == 1:
        img_np = img_np[..., 0]  # (H,W)
    img = Image.fromarray(img_np)
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def base64_to_tensor(b64_str: str, device: torch.device) -> torch.Tensor:
    """
    will Base64 String converted to (C,H,W) Tensor (float32 [0,1]).

    Args:
        b64_str: Base64 encoded image
        device: target device

    Returns:
        torch.Tensor: image tensor (C,H,W)
    """
    img_data = base64.b64decode(b64_str)
    img = Image.open(BytesIO(img_data)).convert("RGB")
    img_np = np.array(img).astype(np.float32) / 255.0  # (H,W,C)
    tensor = torch.from_numpy(img_np).permute(2, 0, 1).to(device)  # (C,H,W)
    return tensor


def pil_to_base64(img: Image.Image, size: tuple = None) -> str:
    """
    will PIL.Image Convert to Base64 string (PNG Format).

    Args:
        img: PIL image
        size: Optional target size (H, W), if specified, first resize

    Returns:
        str: Base64 coded PNG image
    """
    if size is not None:
        # PIL resize The parameters are (W, H), while size Yes (H, W)
        img = img.resize((size[1], size[0]), Image.Resampling.LANCZOS)
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def base64_to_grad_tensor(b64_str: str, device: torch.device) -> torch.Tensor:
    """
    will Base64 coded numpy Array decoded into gradient tensor.

    Server use np.save Save the gradient in the format float32.

    Args:
        b64_str: Base64 coded numpy Bytes
        device: target device

    Returns:
        torch.Tensor: gradient tensor (C,H,W)
    """
    buffer = BytesIO(base64.b64decode(b64_str))
    arr = np.load(buffer)  # (C,H,W) float32
    return torch.from_numpy(arr).to(device)  # (C,H,W)

