"""Loss Functions for Multi-Task Clinical Classification."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class FocalLoss(nn.Module):
    """Multi-class Focal Loss for handling class imbalance in clinical triage."""

    def __init__(
        self,
        alpha: torch.Tensor | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            at = self.alpha.gather(0, targets.data)
            focal_loss = at * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class WeightedCrossEntropyLoss(nn.Module):
    """Weighted Cross Entropy Loss with inverse class frequency weighting."""

    def __init__(self, class_weights: torch.Tensor | None = None):
        super().__init__()
        self.class_weights = class_weights

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weights = (
            self.class_weights.to(inputs.device)
            if self.class_weights is not None
            else None
        )
        return F.cross_entropy(inputs, targets, weight=weights)


class MultiTaskLoss(nn.Module):
    """Multi-task loss combining Triage Severity Loss and Department Loss.

    L_total = w_triage * L_triage + w_dept * L_dept
    """

    def __init__(
        self,
        loss_type: str = "cross_entropy",
        triage_weight: float = 1.0,
        dept_weight: float = 1.0,
        focal_gamma: float = 2.0,
        triage_class_weights: torch.Tensor | None = None,
        dept_class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.triage_weight = triage_weight
        self.dept_weight = dept_weight

        if loss_type == "focal":
            self.triage_loss_fn = FocalLoss(
                alpha=triage_class_weights, gamma=focal_gamma
            )
            self.dept_loss_fn = FocalLoss(alpha=dept_class_weights, gamma=focal_gamma)
        elif loss_type == "weighted_cross_entropy":
            self.triage_loss_fn = WeightedCrossEntropyLoss(
                class_weights=triage_class_weights
            )
            self.dept_loss_fn = WeightedCrossEntropyLoss(
                class_weights=dept_class_weights
            )
        else:
            self.triage_loss_fn = nn.CrossEntropyLoss()
            self.dept_loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        triage_logits: torch.Tensor,
        triage_targets: torch.Tensor,
        dept_logits: torch.Tensor | None = None,
        dept_targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute multi-task loss."""
        loss_triage = self.triage_loss_fn(triage_logits, triage_targets)
        total_loss = self.triage_weight * loss_triage

        loss_metrics = {"loss_triage": float(loss_triage.detach().item())}

        if dept_logits is not None and dept_targets is not None:
            loss_dept = self.dept_loss_fn(dept_logits, dept_targets)
            total_loss += self.dept_weight * loss_dept
            loss_metrics["loss_dept"] = float(loss_dept.detach().item())

        loss_metrics["loss_total"] = float(total_loss.detach().item())
        return total_loss, loss_metrics
