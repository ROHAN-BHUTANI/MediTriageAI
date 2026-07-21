"""Dynamic Clinical Evidence Synthesizer (DCES) for E-PATH-CO-REASON.

Decomposes patient complaints into four latent clinical aspects (Symptom, 
Anatomical, Temporal, and Systemic) to capture distinct clinical pathways.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import torch
import torch.nn as nn

from models.emergent_path_triage.config import EmergentPathTriageConfig
from models.emergent_path_triage.exceptions import InterfaceError
from models.emergent_path_triage.interfaces import BaseClinicalEvidenceSynthesizer
from models.emergent_path_triage.logger import get_logger
from models.emergent_path_triage.types import EvidenceRepresentation

logger = get_logger()


class BasePooler(nn.Module, ABC):
    """Abstract interface defining the contract for token aggregation poolers."""

    @abstractmethod
    def forward(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Aggregate token-level representations into a sequence-level context vector.
        
        Args:
            token_embeddings: Token context vectors.
                Shape: (Batch_Size, Sequence_Length, Hidden_Dimension)
                Dtype: torch.float32
            attention_mask: Active token indicator.
                Shape: (Batch_Size, Sequence_Length)
                Dtype: torch.long or torch.bool
                
        Returns:
            Aggregated context representation.
                Shape: (Batch_Size, Hidden_Dimension)
                Dtype: torch.float32
        """
        raise NotImplementedError


class MaskedMeanPooler(BasePooler):
    """Numerically stable mean pooler computing sequence-level contextual representations.
    
    Robustly handles edge cases such as fully padded sequences and zero-length inputs
    without generating NaN or Inf values.
    """

    def forward(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Perform numerically stable mean pooling over active sequence steps."""
        mask = attention_mask.unsqueeze(-1).float()
        
        # Accumulate token context representations
        summed = (token_embeddings * mask).sum(dim=1)
        
        # Count non-padded tokens per sample
        counts = mask.sum(dim=1)
        
        # Stability Epsilon replacement: where counts is 0, replace with 1.0 to avoid Division-by-Zero
        safe_counts = torch.where(counts == 0.0, torch.ones_like(counts), counts)
        pooled = summed / safe_counts
        
        # Zero-fill outputs for samples that are fully padded (zero active tokens)
        # This guarantees clean zero vectors instead of NaN/Inf values.
        pooled = torch.where(counts == 0.0, torch.zeros_like(pooled), pooled)
        return pooled


class ProjectionBlock(nn.Module):
    """Modular, parameter-isolated MLP projection layer for aspect synthesis.
    
    Guarantees that each clinical aspect maintains its own learnable weights,
    normalization, and activation parameters without sharing representations.
    """

    def __init__(self, hidden_dim: int, latent_dim: int, config: EmergentPathTriageConfig) -> None:
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, latent_dim)
        
        # Parameter-isolated normalization
        if config.dces_normalization == "layernorm":
            self.norm = nn.LayerNorm(latent_dim)
        else:
            self.norm = nn.Identity()
            
        # Parameter-isolated activation
        activation_str = config.dces_activation.lower()
        if activation_str == "gelu":
            self.act = nn.GELU()
        elif activation_str == "relu":
            self.act = nn.ReLU()
        elif activation_str == "silu":
            self.act = nn.SiLU()
        elif activation_str == "tanh":
            self.act = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation function: '{config.dces_activation}'")
            
        self.dropout = nn.Dropout(config.dces_dropout)
        self.linear2 = nn.Linear(latent_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map pooled contextual embeddings to the aspect-specific latent space."""
        return self.linear2(self.dropout(self.act(self.norm(self.linear1(x)))))


class ClinicalEvidenceSynthesizer(BaseClinicalEvidenceSynthesizer):
    """Dynamic Clinical Evidence Synthesizer (DCES).
    
    ============================================================================
    MATH FORMULATION & RATIONALE
    ============================================================================
    Clinical complaints exhibit high diagnostic variance. Decomposing complaints 
    into four clinical aspects allows E-PATH-CO-REASON to evaluate:
      - Symptom: What is the clinical presentation (e.g., "throbbing pain")?
      - Anatomical: Where is it localized (e.g., "left lower abdominal quadrant")?
      - Temporal: How has it progressed (e.g., "gradual onset over 3 days")?
      - Systemic: Are there systemic signs (e.g., "associated with high fever")?
      
    Given token contextual embeddings H in R^{B x L x D} and attention mask M:
      1. Aggregation: x_pool = Pooler(H, M) in R^{B x D}
      2. Projection:  z_aspect = Projection_aspect(x_pool) in R^{B x d}
         for aspect in {symptom, anatomical, temporal, systemic}.
         
    ============================================================================
    COMPUTATIONAL COMPLEXITY
    ============================================================================
    Time Complexity: O(B * L * D) for masked pooling + O(B * D * d + B * d^2) 
      for projection MLP layers.
    Space Complexity: O(B * d) intermediate allocations per forward pass.
    """

    def __init__(self, hidden_dim: int, config: EmergentPathTriageConfig) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.config = config

        # Modular token aggregator abstraction
        self.pooler = MaskedMeanPooler()

        # Reusable Projection Block Factory providing parameter isolation
        self.symptom_proj = ProjectionBlock(hidden_dim, config.latent_dim, config)
        self.anatomical_proj = ProjectionBlock(hidden_dim, config.latent_dim, config)
        self.temporal_proj = ProjectionBlock(hidden_dim, config.latent_dim, config)
        self.systemic_proj = ProjectionBlock(hidden_dim, config.latent_dim, config)
        
        logger.info(
            f"Initialized ClinicalEvidenceSynthesizer with hidden_dim={hidden_dim}, "
            f"latent_dim={config.latent_dim}, activation='{config.dces_activation}', "
            f"normalization='{config.dces_normalization}', dropout={config.dces_dropout}"
        )

    def forward(
        self, 
        token_embeddings: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> EvidenceRepresentation:
        """Synthesize orthogonal clinical aspect representations from transformer embeddings."""
        # 1. Device and dtype validations
        device = next(self.parameters()).device
        
        if not isinstance(token_embeddings, torch.Tensor):
            raise InterfaceError(f"token_embeddings must be a torch.Tensor, got {type(token_embeddings)}")
        if not isinstance(attention_mask, torch.Tensor):
            raise InterfaceError(f"attention_mask must be a torch.Tensor, got {type(attention_mask)}")

        if token_embeddings.device != device:
            raise InterfaceError(
                f"Device mismatch: token_embeddings resides on {token_embeddings.device} "
                f"but module parameters are on {device}"
            )
        if attention_mask.device != device:
            raise InterfaceError(
                f"Device mismatch: attention_mask resides on {attention_mask.device} "
                f"but module parameters are on {device}"
            )

        if token_embeddings.dtype != torch.float32:
            amp_enabled = torch.is_autocast_enabled()
            ckpt_precision = getattr(self.config, "checkpoint_precision_mode", "Unknown")
            raise InterfaceError(
                f"Incorrect dtype: token_embeddings must be torch.float32, got {token_embeddings.dtype}.\n"
                f"Diagnostics:\n"
                f"  Module: ClinicalEvidenceSynthesizer\n"
                f"  Tensor Name: token_embeddings\n"
                f"  Shape: {token_embeddings.shape}\n"
                f"  Device: {token_embeddings.device}\n"
                f"  Expected dtype: torch.float32\n"
                f"  Actual dtype: {token_embeddings.dtype}\n"
                f"  Previous module: XLMRobertaModel (Encoder)\n"
                f"  Current module: ClinicalEvidenceSynthesizer (DCES)\n"
                f"  Checkpoint precision mode: {ckpt_precision}\n"
                f"  AMP enabled status: {amp_enabled}"
            )
        if attention_mask.dtype not in (torch.long, torch.int, torch.bool):
            raise InterfaceError(
                f"Incorrect dtype: attention_mask must be bool or integer type, got {attention_mask.dtype}"
            )

        # 2. Shape validation
        if len(token_embeddings.shape) != 3:
            raise InterfaceError(
                f"token_embeddings must be 3D tensor of shape (Batch, SeqLen, Hidden), got {token_embeddings.shape}"
            )
        if len(attention_mask.shape) != 2:
            raise InterfaceError(
                f"attention_mask must be 2D tensor of shape (Batch, SeqLen), got {attention_mask.shape}"
            )

        batch_size, seq_len, hidden_dim = token_embeddings.shape
        mask_batch, mask_len = attention_mask.shape

        if batch_size != mask_batch:
            raise InterfaceError(
                f"Batch dimension mismatch: token_embeddings has batch size {batch_size} "
                f"but attention_mask has {mask_batch}"
            )
        if seq_len != mask_len:
            raise InterfaceError(
                f"Sequence dimension mismatch: token_embeddings has sequence length {seq_len} "
                f"but attention_mask has {mask_len}"
            )
        if hidden_dim != self.hidden_dim:
            raise InterfaceError(
                f"Hidden dimension mismatch: token_embeddings has hidden dimension {hidden_dim} "
                f"but expected {self.hidden_dim}"
            )

        # 3. Dynamic sequence-level aggregation
        pooled = self.pooler(token_embeddings, attention_mask)

        # 4. Aspect projections mapping
        z_symptom = self.symptom_proj(pooled)
        z_anatomical = self.anatomical_proj(pooled)
        z_temporal = self.temporal_proj(pooled)
        z_systemic = self.systemic_proj(pooled)

        # Enforce clean zero output for fully padded sequences (zero active tokens)
        # to prevent bias leakage or LayerNorm shifting from creating non-zero outputs
        counts = attention_mask.sum(dim=1, keepdim=True)
        z_symptom = torch.where(counts == 0, torch.zeros_like(z_symptom), z_symptom)
        z_anatomical = torch.where(counts == 0, torch.zeros_like(z_anatomical), z_anatomical)
        z_temporal = torch.where(counts == 0, torch.zeros_like(z_temporal), z_temporal)
        z_systemic = torch.where(counts == 0, torch.zeros_like(z_systemic), z_systemic)

        return EvidenceRepresentation(
            symptom=z_symptom,
            anatomical=z_anatomical,
            temporal=z_temporal,
            systemic=z_systemic
        )
