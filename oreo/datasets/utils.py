import math
import torch


def _normalize_fovy_to_batch(
    fovy_deg: torch.Tensor,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
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
    """Constructs a batch perspective projection matrix.

    parameters:
    - fovy_deg: [] | [B]
    - aspect: Scalar aspect ratio
    - znear, zfar: Scalar near and far clipping planes
    - dtype, device: Tensor types and devices
    - batch_size: expected batch size (used to convert scalar fovy expanded to [B])

    Return:
    - proj: [B,4,4]
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
    """According to w2c Constructed with projection matrix MVP.

    Support two w2c Shape:
    - [B,4,4]
    - [B,V,4,4]

    request proj The shape is [B,4,4].
    Return with w2c same dimension mvp.
    """
    if w2c.dim() == 3:
        mvp = torch.matmul(proj, w2c)  # [B,4,4]
        return mvp  # [B,4,4]
    if w2c.dim() == 4:
        mvp = torch.matmul(proj[:, None, ...], w2c)  # [B,V,4,4]
        return mvp  # [B,V,4,4]
    raise ValueError("w2c must be [B,4,4] or [B,V,4,4]")


def build_w2c_from_c2w(c2w: torch.Tensor) -> torch.Tensor:
    """from c2w Calculate w2c, support [B,4,4] or [B,V,4,4].

    Note: Do not assume strict orthogonality, use matrix inversion directly.
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

