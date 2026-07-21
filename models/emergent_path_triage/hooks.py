"""Hooks for integrating E-PATH-CO-REASON with training and evaluation scripts."""

from __future__ import annotations

from typing import Any
import torch
import torch.nn as nn


def apply_loss_hook(
    model: nn.Module,
    specialist_logits: torch.Tensor,
    severity_logits: torch.Tensor,
    labels_specialist: torch.Tensor,
    labels_severity: torch.Tensor,
    loss_fn: nn.Module,
) -> dict[str, torch.Tensor]:
    """Intercept loss computation and delegate to E-PATH-CO-REASON custom loss solver.
    
    If the model does not support a custom loss, defaults to standard joint loss evaluation.
    """
    if hasattr(model, "compute_loss"):
        return model.compute_loss(
            specialist_logits,
            severity_logits,
            labels_specialist,
            labels_severity,
            loss_fn,
        )
    return loss_fn(specialist_logits, severity_logits, labels_specialist, labels_severity)
