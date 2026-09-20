"""Loss Management and gradient calculation"""
import torch
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple, Union


# =====================================================================
# uniformly weighted Loss(True Loss mode, can be directly backward)
# =====================================================================

def apply_gradient_loss(
    stu: torch.Tensor,
    tea: torch.Tensor,
    clean: torch.Tensor,
    weight_mode: str = "uniform",
    t_norm: Optional[float] = None,
    dim: Optional[Union[int, Tuple[int, ...]]] = None,
    eps: float = 1e-2,
) -> torch.Tensor:
    """
    uniform weighting loss(True Loss mode, can be directly backward).

    Supports three weight modes:
    - uniform: MSE(stu, tea)
    - t: time step weighting MSE
    - ada: Adaptive Normalization (DMD eq.8 style)

    Args:
        stu: Student model prediction (requires gradient backpass)
        tea: Teacher model prediction (no gradient)
        clean: true clean data (for ada pattern normalizer)
        weight_mode: "uniform" | "t" | "ada"
        t_norm: Normalized time step (t mode required)
        dim: ada The normalized dimension of the pattern (None Represents the global)
        eps: ada pattern epsilon

    Returns:
        loss: true loss, can be directly backward
    """
    tea = tea.detach()  # Ensure teachers have no gradient

    if weight_mode == "uniform":
        # Standard MSE
        return F.mse_loss(stu.float(), tea.float())

    elif weight_mode == "t":
        # time step weighting MSE
        if t_norm is None:
            raise ValueError("weight_mode='t' Need to provide t_norm")
        diff = (stu - tea).float()  # [...]
        weighted_diff = diff * t_norm
        return (weighted_diff ** 2).mean()

    elif weight_mode == "ada":
        # Adaptive normalization: first calculate the target gradient, then normalize
        # target gradient = stu - tea
        grad_raw = (stu - tea).detach().float()  # No gradient, used for calculation normalizer

        # normalizer = |clean - tea|.mean(dim) + eps(or |stu - tea|.mean())
        with torch.no_grad():
            if dim is None:
                normalizer = grad_raw.abs().mean() + eps  # scalar
            else:
                normalizer = grad_raw.abs().mean(dim=dim, keepdim=True) + eps
            grad_normalized = grad_raw / normalizer

        # structure loss, make ∂loss/∂stu = grad_normalized
        return (stu.float() * grad_normalized).mean()

    else:
        raise ValueError(f"Unknown weight_mode: {weight_mode}. Use 'uniform', 't', or 'ada'.")


# =====================================================================
# LossDict - unify Loss management
# =====================================================================

class LossDict:
    """
    unify loss Management: accumulation, weighting, log generation.

    Usage:
        losses = LossDict(device="cuda:0")  # Specify target device
        losses.add("ssim", loss_ssim, weight=cfg.ssim_weight)
        losses.add("lpips", loss_lpips, weight=cfg.lpips_weight)

        total = losses.total()           # Automatic summation (all loss Moved to the same device)
        logs = losses.to_logs()          # {"loss/ssim": ..., "loss/lpips": ..., "loss/total": ...}
    """

    def __init__(self, device: torch.device = None):
        self._items: Dict[str, torch.Tensor] = {}  # weighted loss
        self._raw: Dict[str, torch.Tensor] = {}    # original loss(for logging)
        self._device = device  # target device for unification tensor location

    def add(
        self,
        name: str,
        loss: Optional[torch.Tensor],
        weight: float = 1.0,
    ) -> "LossDict":
        """
        add loss item.

        Args:
            name: loss Name (such as "ssim", "lpips")
            loss: loss tensor or None
            weight: weight (default 1.0, indicating that the weights have been applied externally)

        Returns:
            self(Supports chain calls)
        """
        if loss is None or weight <= 0:
            return self

        # Move to target device (if specified)
        if self._device is not None and loss.device != self._device:
            loss = loss.to(self._device)

        weighted = loss * weight if weight != 1.0 else loss
        self._items[name] = weighted
        self._raw[name] = loss
        return self

    def total(self) -> torch.Tensor:
        """Calculate weighting loss sum"""
        if not self._items:
            device = self._device if self._device else "cpu"
            return torch.tensor(0.0, device=device)

        return sum(self._items.values())

    def to_logs(self, prefix: str = "loss/") -> Dict[str, torch.Tensor]:
        """
        Generate log dictionary.

        Args:
            prefix: key prefix (default "loss/")

        Returns:
            dict: {"loss/ssim": tensor, "loss/lpips": tensor, "loss/total": tensor}
        """
        logs = {}
        for name, val in self._raw.items():
            logs[f"{prefix}{name}"] = val.detach()

        if self._items:
            logs[f"{prefix}total"] = self.total().detach()

        return logs

    def __bool__(self) -> bool:
        """are there any loss"""
        return bool(self._items)
