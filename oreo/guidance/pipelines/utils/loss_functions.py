"""
Loss helpers and multi-timestep aggregation for FlowEdit / contrastive training.

Design:
1. Single-step loss on (src, target)
2. Multi-step reduce: final / mean / weighted / inv_weighted
3. LossMixin on trackers
4. Orthogonal axes: ada normalize vs reduce mode
"""

from typing import List, Literal, Optional
import torch
import torch.nn.functional as F


# =============================================================================
# type definition
# =============================================================================

ReduceMode = Literal["final", "mean", "weighted", "inv_weighted"]


# =============================================================================
# Utility function
# =============================================================================

def reduce_losses(losses: List[torch.Tensor], mode: ReduceMode = "final") -> torch.Tensor:
    """
    Aggregate a list of scalar step losses.
    
    Modes: final | mean | weighted(1/k) | inv_weighted(k/K).
    """
    if len(losses) == 0:
        raise ValueError("losses list is empty")

    if mode == "final":
        return losses[-1]  # scalar

    losses_t = torch.stack(losses)  # [K]
    K = len(losses_t)  # number of steps

    if mode == "mean":
        return losses_t.mean()  # scalar

    elif mode == "weighted":
        # 1/k Weighted:[1, 1/2, 1/3, ..., 1/K]
        w = 1.0 / torch.arange(1, K + 1, device=losses_t.device, dtype=losses_t.dtype)  # [K]

    elif mode == "inv_weighted":
        # k/K Weighted:[1/K, 2/K, ..., 1]
        w = torch.arange(1, K + 1, device=losses_t.device, dtype=losses_t.dtype) / K  # [K]

    else:
        raise ValueError(f"Unknown mode: {mode}. Choose from: final, mean, weighted, inv_weighted")

    w = w / w.sum()  # [K] normalization
    return (losses_t * w).sum()  # scalar


def normalize_grad(
    grad: torch.Tensor,
    src: torch.Tensor,
    pos: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Ada normalize: grad / (mean|src-pos| + eps). Uses L1 distance to the attractor as scale."""
    # Scale by mean L1(src, pos) so step size adapts to edit magnitude.
    normalizer = torch.abs(src - pos).detach().mean(dim=(1, 2), keepdim=True) + eps  # [B, 1, 1]
    return grad / normalizer  # [B, seq, C]


# =============================================================================
# single step Loss(basic building blocks)
# =============================================================================

def mse_loss_step(
    src: torch.Tensor,
    target: torch.Tensor,
    ada: bool = False,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Single-step MSE(src, target), optional ada. Shapes: [B,seq,C]."""
    if ada:
        grad = (src - target).detach().float()  # [B, seq, C]
        # MSE Scenario:target That's it pos
        grad_norm = normalize_grad(grad, src.float(), target.detach().float(), eps)  # [B, seq, C]
        return (src.float() * grad_norm).mean()  # scalar
    else:
        return F.mse_loss(
            src.float(),  # [B, seq, C]
            target.detach().float()  # [B, seq, C]
        )  # scalar


def contrastive_loss_step(
    src: torch.Tensor,
    pos: torch.Tensor,
    neg: torch.Tensor,
    ada: bool = False,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Single-step contrastive: MSE(src,pos) - MSE(src,neg) — pull to pos, push from neg."""
    # Pull src toward pos, push away from neg in latent space.
    if ada:
        grad = (neg - pos).detach().float()  # [B, seq, C] Direction:neg - pos
        grad_norm = normalize_grad(grad, src.float(), pos.detach().float(), eps)  # [B, seq, C]
        return (src.float() * grad_norm).mean()  # scalar
    else:
        loss_pos = F.mse_loss(
            src.float(),  # [B, seq, C]
            pos.detach().float()  # [B, seq, C]
        )  # scalar
        loss_neg = F.mse_loss(
            src.float(),  # [B, seq, C]
            neg.detach().float()  # [B, seq, C]
        )  # scalar
        return loss_pos - loss_neg  # scalar



# =============================================================================
# multiple steps Loss(combined interface)
# =============================================================================

def mse_loss(
    src: torch.Tensor,
    targets: List[torch.Tensor],
    reduce: ReduceMode = "final",
    ada: bool = False,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Multi-step MSE with reduce_mode."""
    if len(targets) == 0:
        return torch.tensor(0.0, device=src.device, requires_grad=True)  # scalar

    losses = [
        mse_loss_step(src, t, ada=ada, eps=eps)  # scalar
        for t in targets
    ]
    return reduce_losses(losses, reduce)  # scalar


def csd_loss(
    src: torch.Tensor,
    x0_pos_list: List[torch.Tensor],
    x0_neg_list: List[torch.Tensor],
    reduce: ReduceMode = "final",
    ada: bool = False,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Multi-step CSD: MSE(src,x0_pos) - MSE(src,x0_neg)."""
    if len(x0_pos_list) == 0:
        return torch.tensor(0.0, device=src.device, requires_grad=True)  # scalar

    losses = [
        contrastive_loss_step(src, pos, neg, ada=ada, eps=eps)  # scalar
        for pos, neg in zip(x0_pos_list, x0_neg_list)
    ]
    return reduce_losses(losses, reduce)  # scalar


# =============================================================================
# LossMixin - Tracker Used loss Calculate Mixin
# =============================================================================

class LossMixin:
    """Mixin exposing mse / contrastive / csd helpers on a tracker."""

    # Subclasses need to define these fields
    x0_preds: List[torch.Tensor]   # MSE target(FlowEdit Zhongwei z_edit)
    x0_pos: List[torch.Tensor]     # CSD Positive sample (high CFG x0 predict, attract)
    x0_neg: List[torch.Tensor]     # CSD Negative samples (low CFG x0 prediction, rejection)

    def compute_mse_loss(
        self,
        src: torch.Tensor,
        ada: bool = False,
        eps: float = 1e-4,
        reduce: ReduceMode = "mean",
    ) -> torch.Tensor:
        """
        Calculate MSE Loss: MSE(src, x0_preds)

        Args:
            src: [B, seq, C] There is a gradient
            ada: Whether to use adaptive normalization
            eps: ada pattern epsilon
            reduce: Aggregation method

        Returns:
            scalar loss
        """
        return mse_loss(src, self.x0_preds, reduce=reduce, ada=ada, eps=eps)

    def compute_csd_loss(
        self,
        src: torch.Tensor,
        ada: bool = False,
        eps: float = 1e-4,
        reduce: ReduceMode = "mean",
    ) -> torch.Tensor:
        """
        Calculate CSD Loss:MSE(src, x0_pos) - MSE(src, x0_neg)

        Args:
            src: [B, seq, C] There is a gradient
            ada: Whether to use adaptive normalization
            eps: ada pattern epsilon
            reduce: Aggregation method

        Returns:
            scalar loss
        """
        return csd_loss(src, self.x0_pos, self.x0_neg, reduce=reduce, ada=ada, eps=eps)

    def loss(
        self,
        src: torch.Tensor,
        mse_weight: float = 0.0,
        csd_weight: float = 1.0,
        ada: bool = False,
        eps: float = 1e-4,
        reduce: ReduceMode = "mean",
    ) -> torch.Tensor:
        """
        Compute Mix Loss.

        Loss = mse_weight * MSE(src, x0_preds)
             + csd_weight * CSD(src, x0_pos, x0_neg)

        Args:
            src: [B, seq, C] source with gradient latent
            mse_weight: MSE loss weight
            csd_weight: CSD loss weight
            ada: Whether to use adaptive normalization
            eps: ada pattern epsilon
            reduce: Aggregation method

        Returns:
            scalar loss
        """
        total_loss = torch.tensor(0.0, device=src.device, dtype=src.dtype)

        if mse_weight > 0 and self.x0_preds:
            loss_mse = self.compute_mse_loss(src, ada=ada, eps=eps, reduce=reduce)
            total_loss = total_loss + mse_weight * loss_mse

        if csd_weight > 0 and self.x0_pos and self.x0_neg:
            loss_csd = self.compute_csd_loss(src, ada=ada, eps=eps, reduce=reduce)
            total_loss = total_loss + csd_weight * loss_csd

        return total_loss
