"""Dual-task Prediction Heads implementation for E-PATH-CO-REASON."""

from __future__ import annotations

import torch
from torch import nn

from models.emergent_path_triage.config import EmergentPathTriageConfig
from models.emergent_path_triage.exceptions import InterfaceError
from models.emergent_path_triage.logger import get_logger

logger = get_logger()


class PredictionHead(nn.Module):
    """Reusable prediction classifier head.

    Transforms the final reasoning state representation into task-specific logits
    using a multi-layer perceptron with normalization, activation, and dropout.

    ============================================================================
    MATH FORMULATION & RATIONALE
    ============================================================================
    For a final latent reasoning representation h_M in R^{B x d}:
      1. Pre-Normalization: h_norm = LayerNorm(h_M) in R^{B x d}
      2. Hidden Layer:      h_hidden = Act(Linear1(h_norm)) in R^{B x H_head}
      3. Task Classification Logits: y = Linear2(Dropout(h_hidden)) in R^{B x C}
    where:
      - C is the number of target classes (13 for Specialist, 5 for Severity).
      - H_head is the head hidden dimension (`head_hidden_dim`).

    ============================================================================
    COMPUTATIONAL COMPLEXITY
    ============================================================================
    Time Complexity: O(B * (d * H_head + H_head * C))
    Space Complexity: O(B * C) memory allocation for logits.
    """

    def __init__(
        self, latent_dim: int, output_dim: int, config: EmergentPathTriageConfig
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.config = config

        self.norm = nn.LayerNorm(latent_dim)

        # Set up Activation Function
        activation_str = config.head_activation.lower()
        if activation_str == "gelu":
            self.act = nn.GELU()
        elif activation_str == "relu":
            self.act = nn.ReLU()
        elif activation_str == "silu":
            self.act = nn.SiLU()
        elif activation_str == "tanh":
            self.act = nn.Tanh()
        else:
            raise ValueError(
                f"Unsupported activation function: '{config.head_activation}'"
            )

        self.fc1 = nn.Linear(latent_dim, config.head_hidden_dim)
        self.dropout = nn.Dropout(config.head_dropout)
        self.fc2 = nn.Linear(config.head_hidden_dim, output_dim)

        logger.info(
            f"Initialized PredictionHead mapping {latent_dim} -> {output_dim} "
            f"via head_hidden_dim={config.head_hidden_dim}, act='{activation_str}', "
            f"dropout={config.head_dropout}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute logits from latent reasoning representation."""
        # 1. Verification and validations
        device = next(self.parameters()).device

        if not isinstance(x, torch.Tensor):
            raise InterfaceError(f"Input must be a torch.Tensor, got {type(x)}")

        if x.device != device:
            raise InterfaceError(
                f"Device mismatch: Input resides on {x.device} "
                f"but head parameters are on {device}"
            )

        if x.dtype != torch.float32:
            raise InterfaceError(
                f"Incorrect dtype: Input must be torch.float32, got {x.dtype}"
            )

        if len(x.shape) != 2:
            raise InterfaceError(
                f"Input must be 2D tensor of shape (Batch, Latent_Dim), got {x.shape}"
            )

        batch_size, latent_dim = x.shape
        if latent_dim != self.latent_dim:
            raise InterfaceError(
                f"Latent dimension mismatch: Input has dimension {latent_dim} "
                f"but expected {self.latent_dim}"
            )

        # 2. MLP Forward Pass
        normed = self.norm(x)
        hidden = self.act(self.fc1(normed))
        logits = self.fc2(self.dropout(hidden))

        return logits
