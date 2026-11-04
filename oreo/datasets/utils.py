"""
Dataset geometry helpers for Trellis cameras: perspective projection, MVP, and W2C/C2W.

Used by oreo.datasets.trellis when building intrinsics / extrinsics batches for rendering.
"""

import math
import torch


def _normalize_fovy_to_batch(
    fovy_deg: torch.Tensor,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Normalize FOV-Y to a length-B tensor on the given device/dtype."""
    # will fovy unified as [B]
    if not torch.is_tensor(fovy_deg):
        fovy_t = torch.full((batch_size,), float(fovy_deg), dtype=dtype, device=device)  # [B]
    else:
        if fovy_deg.dim() == 0:
            fovy_t = torch.full((batch_size,), float(fovy_deg.item()), dtype=dtype, device=device)  # [B]
        else:
            fovy_t = fovy_deg.to(device=device, dtype=dtype).reshape(-1)  # [B]
    return fovy_t  # [B]


def build_perspective_matrix(
    fovy_deg: torch.Tensor,
    aspect: float,
    znear: float,
    zfar: float,
    dtype: torch.dtype,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """
    Build a batch of perspective projection matrices.
    
    Args:
    - fovy_deg: [] or [B] vertical FOV in degrees
    - aspect: scalar aspect ratio
    - znear, zfar: near/far planes
    Returns: [B,4,4] OpenGL-style projection.
    """
    fovy_t = _normalize_fovy_to_batch(fovy_deg, batch_size, dtype, device)  # [B]

    f = 1.0 / torch.tan((fovy_t * math.pi / 180.0) * 0.5)  # [B]
    A = torch.tensor(aspect, dtype=dtype, device=device)  # []
    zn = torch.tensor(znear, dtype=dtype, device=device)  # []
    zf = torch.tensor(zfar, dtype=dtype, device=device)  # []

    proj = torch.zeros((batch_size, 4, 4), dtype=dtype, device=device)  # [B,4,4]
    proj[:, 0, 0] = f / A  # [B]
    # with nvdiffrast Agreed alignment:Y The axis is negative to prevent the rendering result from being upside down.
    proj[:, 1, 1] = -f  # [B]
    proj[:, 2, 2] = (zf + zn) / (zn - zf)  # [B]
    proj[:, 2, 3] = (2.0 * zf * zn) / (zn - zf)  # [B]
    proj[:, 3, 2] = -1.0  # [B]
    return proj  # [B,4,4]


def build_mvp_from_w2c(
    w2c: torch.Tensor,
    proj: torch.Tensor,
) -> torch.Tensor:
    """
    Build MVP from world-to-camera and projection matrices.
    
    Supports w2c shapes [B,4,4] or [B,V,4,4]; proj must broadcast accordingly.
    """
    if w2c.dim() == 3:
        mvp = torch.matmul(proj, w2c)  # [B,4,4]
        return mvp  # [B,4,4]
    if w2c.dim() == 4:
        mvp = torch.matmul(proj[:, None, ...], w2c)  # [B,V,4,4]
        return mvp  # [B,V,4,4]
    raise ValueError("w2c must be [B,4,4] or [B,V,4,4]")


def build_w2c_from_c2w(c2w: torch.Tensor) -> torch.Tensor:
    """
    Invert c2w to w2c for [B,4,4] or [B,V,4,4].
    Uses general matrix inverse (no orthogonality assumption).
    """
    if c2w.dim() == 3:
        w2c = torch.inverse(c2w)  # [B,4,4]
        return w2c  # [B,4,4]
    if c2w.dim() == 4:
        B, V = c2w.shape[:2]  # [], []
        c2w_flat = c2w.reshape(B * V, 4, 4)  # [B*V,4,4]
        w2c_flat = torch.inverse(c2w_flat)  # [B*V,4,4]
        w2c = w2c_flat.reshape(B, V, 4, 4)  # [B,V,4,4]
        return w2c  # [B,V,4,4]
    raise ValueError("c2w must be [B,4,4] or [B,V,4,4]")

