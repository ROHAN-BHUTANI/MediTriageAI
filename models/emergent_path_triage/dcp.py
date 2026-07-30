"""Dynamic Consistency Projection (DCP) implementation for E-PATH-CO-REASON."""

from __future__ import annotations

import torch
from torch import nn

from models.emergent_path_triage.config import EmergentPathTriageConfig
from models.emergent_path_triage.exceptions import InterfaceError
from models.emergent_path_triage.interfaces import BaseConsistencyProjection
from models.emergent_path_triage.logger import get_logger

logger = get_logger()


class DynamicConsistencyProjection(BaseConsistencyProjection):
    """Dynamic Consistency Projection (DCP).

    Projects the latent reasoning path states and final predictions (specialist
    and severity logits) into a shared urgency space. This acts as a regularizer
    that binds downstream classifications to the executed reasoning path.

    ============================================================================
    MATH FORMULATION & RATIONALE
    ============================================================================
    Rather than letting classifier heads run completely disjointly from the path
    trajectory, DCP aligns:
      1. Path Trajectory Representation h_M in R^{B x d}
      2. Joint Logit Predictions [y_spec; y_sev] in R^{B x (13 + 5)}

    We map both into the shared urgency space (dimension 5):
      h_proj = Linear_reasoning(h_M) in R^{B x 5}
      y_proj = Linear_logits([y_spec; y_sev]) in R^{B x 5}

    The alignment error (Consistency Loss) is calculated as:
      L_cons = Mean(|| h_proj - y_proj ||_2^2)

    ============================================================================
    COMPUTATIONAL COMPLEXITY
    ============================================================================
    Time Complexity: O(B * (d * 5 + 18 * 5))
    Space Complexity: O(B * 5) memory allocation for projected vectors.
    """

    def __init__(self, latent_dim: int, config: EmergentPathTriageConfig) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.config = config

        # Projections to shared urgency space (dimension 5 matching severity)
        self.reasoning_proj = nn.Linear(latent_dim, 5, bias=False)
        self.logits_proj = nn.Linear(18, 5, bias=False)

        logger.info(
            f"Initialized DynamicConsistencyProjection with latent_dim={latent_dim}, "
            f"urgency_dim=5"
        )

    def forward(
        self, specialist_state: torch.Tensor, severity_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project specialist state (reasoning) and severity state (logits) to urgency space.

        Args:
            specialist_state: Final latent reasoning state representation of shape (Batch, Latent_Dim).
            severity_state: Concatenated predictions logits [spec_logits; sev_logits] of shape (Batch, 18).

        Returns:
            projected_reasoning: Projected trajectory state in urgency space: (Batch, 5).
            projected_predictions: Projected predictions state in urgency space: (Batch, 5).
        """
        # 1. Verification and validations
        device = next(self.parameters()).device

        if not isinstance(specialist_state, torch.Tensor):
            raise InterfaceError(
                f"specialist_state must be a torch.Tensor, got {type(specialist_state)}"
            )
        if not isinstance(severity_state, torch.Tensor):
            raise InterfaceError(
                f"severity_state must be a torch.Tensor, got {type(severity_state)}"
            )

        if specialist_state.device != device:
            raise InterfaceError(
                f"Device mismatch: specialist_state resides on {specialist_state.device} "
                f"but DCP parameters are on {device}"
            )
        if severity_state.device != device:
            raise InterfaceError(
                f"Device mismatch: severity_state resides on {severity_state.device} "
                f"but DCP parameters are on {device}"
            )

        if specialist_state.dtype != torch.float32:
            raise InterfaceError(
                f"Incorrect dtype: specialist_state must be torch.float32, got {specialist_state.dtype}"
            )
        if severity_state.dtype != torch.float32:
            raise InterfaceError(
                f"Incorrect dtype: severity_state must be torch.float32, got {severity_state.dtype}"
            )

        # Shape validation
        if len(specialist_state.shape) != 2:
            raise InterfaceError(
                f"specialist_state must be a 2D tensor, got shape {specialist_state.shape}"
            )
        if len(severity_state.shape) != 2:
            raise InterfaceError(
                f"severity_state must be a 2D tensor, got shape {severity_state.shape}"
            )

        batch_size, latent_dim = specialist_state.shape
        if latent_dim != self.latent_dim:
            raise InterfaceError(
                f"Dimension mismatch: specialist_state has dimension {latent_dim} "
                f"but expected {self.latent_dim}"
            )

        batch_size_sev, logits_dim = severity_state.shape
        if batch_size_sev != batch_size:
            raise InterfaceError(
                f"Batch size mismatch: specialist_state has batch size {batch_size} "
                f"but severity_state has batch size {batch_size_sev}"
            )
        if logits_dim != 18:
            raise InterfaceError(
                f"Dimension mismatch: severity_state must have dimension 18 "
                f"([specialist_logits; severity_logits]), got {logits_dim}"
            )

        # 2. Urgency Projections
        projected_reasoning = self.reasoning_proj(specialist_state)
        projected_predictions = self.logits_proj(severity_state)

        return projected_reasoning, projected_predictions
