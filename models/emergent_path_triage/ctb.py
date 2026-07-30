"""Clinical Thought Block (CTB) implementation for E-PATH-CO-REASON."""

from __future__ import annotations

import torch
from torch import nn

from models.emergent_path_triage.config import EmergentPathTriageConfig
from models.emergent_path_triage.exceptions import InterfaceError
from models.emergent_path_triage.interfaces import BaseClinicalThoughtBlock
from models.emergent_path_triage.logger import get_logger

logger = get_logger()


class ClinicalThoughtBlock(BaseClinicalThoughtBlock):
    """Clinical Thought Block (CTB).

    Acts as a parameter-isolated reasoning node processing contextual latent states.
    Uses pre-normalization feed-forward transformer structure with residual connections.

    ============================================================================
    MATH FORMULATION & RATIONALE
    ============================================================================
    Rather than pre-defining clinical functions (e.g., severity mapping vs symptom
    integration), E-PATH-CO-REASON delegates updates to generic CTBs.
    For an input latent state x in R^{B x d}:
      1. Pre-Normalization: x_norm = LayerNorm(x) in R^{B x d}
      2. Projection Block:  x_ffn = Linear2(Dropout(Act(Linear1(x_norm)))) in R^{B x d}
      3. Residual Mapping:   y = x + x_ffn in R^{B x d}

    This maps updates continuously while preserving the size and mapping space (d).

    ============================================================================
    COMPUTATIONAL COMPLEXITY
    ============================================================================
    Time Complexity: O(B * d * H_ctb) where B=Batch, d=LatentDim, H_ctb=ctb_hidden_dim.
    Space Complexity: O(B * d) memory allocation.
    """

    def __init__(self, latent_dim: int, config: EmergentPathTriageConfig) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.config = config

        # Set up Layer Normalization
        if config.ctb_normalization == "layernorm":
            self.norm = nn.LayerNorm(latent_dim)
        else:
            self.norm = nn.Identity()

        # Set up Activation Function
        activation_str = config.ctb_activation.lower()
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
                f"Unsupported activation function: '{config.ctb_activation}'"
            )

        # Transformer FFN layers
        self.linear1 = nn.Linear(latent_dim, config.ctb_hidden_dim)
        self.dropout = nn.Dropout(config.ctb_dropout)
        self.linear2 = nn.Linear(config.ctb_hidden_dim, latent_dim)

        logger.info(
            f"Initialized ClinicalThoughtBlock with latent_dim={latent_dim}, "
            f"ctb_hidden_dim={config.ctb_hidden_dim}, activation='{activation_str}', "
            f"normalization='{config.ctb_normalization}', dropout={config.ctb_dropout}"
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Perform clinical thought update on latent embeddings."""
        # 1. Device and dtype validations
        device = next(self.parameters()).device

        if not isinstance(state, torch.Tensor):
            raise InterfaceError(f"state must be a torch.Tensor, got {type(state)}")

        if state.device != device:
            raise InterfaceError(
                f"Device mismatch: state resides on {state.device} "
                f"but thought block parameters are on {device}"
            )

        if state.dtype != torch.float32:
            raise InterfaceError(
                f"Incorrect dtype: state must be torch.float32, got {state.dtype}"
            )

        # 2. Shape validation
        if len(state.shape) != 2:
            raise InterfaceError(
                f"state must be a 2D tensor of shape (Batch, Latent_Dim), got {state.shape}"
            )

        batch_size, latent_dim = state.shape
        if latent_dim != self.latent_dim:
            raise InterfaceError(
                f"Latent dimension mismatch: state has dimension {latent_dim} "
                f"but expected {self.latent_dim}"
            )

        # 3. Pre-normalization and residual FFN mapping
        normed = self.norm(state)
        projected = self.linear2(self.dropout(self.act(self.linear1(normed))))

        return state + projected
