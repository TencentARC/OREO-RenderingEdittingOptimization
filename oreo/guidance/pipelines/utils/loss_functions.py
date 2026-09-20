"""
Loss Compute function and time step sampling tools.

Design principles:
1. single step Loss: Calculate a single (src, target) Right loss
2. Multi-step aggregation: combine multiple loss aggregate into one
3. LossMixin: for Tracker Provide a unified loss computing power
4. Time step sampling: supports multi-time step sampling (MTS)

Two orthogonal dimensions:
- Normalization method:ada=False(original MSE)/ ada=True(normalized gradient)
- Aggregation method:final / mean / weighted / inv_weighted
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
    Polymerization multiple steps loss.

    Args:
        losses: List of scalar losses
        mode:
            - "final": Only take the last one
            - "mean": uniform average
            - "weighted": 1/k Weighted (large in the early stage)
            - "inv_weighted": k/K Weighted (larger in the later stage)

    Returns:
        scalar loss
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
    """
    Ada Normalization:grad / (|src - pos|.mean + eps)

    Use src Arrive pos(to attract a target) L1 The mean distance is used as a normalization factor.

    Args:
        grad: [B, seq, C] gradient direction (usually neg - pos or src - target)
        src: [B, seq, C] source latent(to be optimized)
        pos: [B, seq, C] Positive sample (attracting target)
        eps: numerically stable epsilon

    Returns:
        [B, seq, C] Normalized gradient
    """
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
    """
    single step MSE Loss.

    Args:
        src: [B, seq, C] source latent(with gradient)
        target: [B, seq, C] target latent(Also pos)
        ada: Whether to use adaptive normalization
        eps: ada pattern epsilon

    Returns:
        scalar loss
    """
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
    """
    Single step comparison Loss:MSE(src, pos) - MSE(src, neg)

    let src towards pos come closer, stay away neg.

    Args:
        src: [B, seq, C] source latent(with gradient)
        pos: [B, seq, C] Positive sample (attract)
        neg: [B, seq, C] Negative samples (exclusion)
        ada: Whether to use adaptive normalization
        eps: ada pattern epsilon

    Returns:
        scalar loss
    """
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
    """
    multiple steps MSE Loss.

    Args:
        src: [B, seq, C] source latent(with gradient)
        targets: List[[B, seq, C]] target latent list
        reduce: Aggregation method "final" | "mean" | "weighted" | "inv_weighted"
        ada: Whether to use adaptive normalization
        eps: ada pattern epsilon

    Returns:
        scalar loss
    """
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
    """
    multiple steps CSD Loss:MSE(src, x0_pos) - MSE(src, x0_neg)

    Args:
        src: [B, seq, C] source latent(with gradient)
        x0_pos_list: List[[B, seq, C]] CSD Positive sample list (high CFG x0 predict, attract)
        x0_neg_list: List[[B, seq, C]] CSD Negative sample list (low CFG x0 prediction, rejection)
        reduce: Aggregation method "final" | "mean" | "weighted" | "inv_weighted"
        ada: Whether to use adaptive normalization
        eps: ada pattern epsilon

    Returns:
        scalar loss
    """
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
    """
    Loss Calculate Mixin.

    for Tracker Provide a unified loss Computing power:
    - MSE Loss: MSE(src, x0_preds)
    - CSD Loss: MSE(src, x0_pos) - MSE(src, x0_neg)
    - mix Loss: mse_weight * MSE + csd_weight * CSD

    Subclasses need to provide:
    - x0_preds: List[Tensor] Forecast x0 list(MSE target,FlowEdit Zhongwei z_edit)
    - x0_pos: List[Tensor] CSD Positive sample list (high CFG x0 predict, attract)
    - x0_neg: List[Tensor] CSD Negative sample list (low CFG x0 prediction, rejection)
    """

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
