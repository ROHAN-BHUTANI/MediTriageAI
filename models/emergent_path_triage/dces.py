"""Dynamic Clinical Evidence Synthesizer (DCES) for E-PATH-CO-REASON.

Decomposes patient complaints into four latent clinical aspects (Symptom,
Anatomical, Temporal, and Systemic) to capture distinct clinical pathways.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn

from models.emergent_path_triage.config import EmergentPathTriageConfig
from models.emergent_path_triage.exceptions import InterfaceError
from models.emergent_path_triage.interfaces import BaseClinicalEvidenceSynthesizer
from models.emergent_path_triage.logger import get_logger
from models.emergent_path_triage.types import (
    EvidenceAttentionRecorder,
    EvidenceReasoningTrace,
    EvidenceRepresentation,
)

logger = get_logger()


class BasePooler(nn.Module, ABC):
    """Abstract interface defining the contract for token aggregation poolers."""

    @abstractmethod
    def forward(
        self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
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

    def forward(
        self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
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

    def __init__(
        self, hidden_dim: int, latent_dim: int, config: EmergentPathTriageConfig
    ) -> None:
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
            raise ValueError(
                f"Unsupported activation function: '{config.dces_activation}'"
            )

        self.dropout = nn.Dropout(config.dces_dropout)
        self.linear2 = nn.Linear(latent_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map pooled contextual embeddings to the aspect-specific latent space."""
        return self.linear2(self.dropout(self.act(self.norm(self.linear1(x)))))


class BaseEvidenceFusion(nn.Module, ABC):
    """Abstract interface defining the contract for evidence fusion modules."""

    @abstractmethod
    def forward(
        self,
        z_symptom: torch.Tensor,
        z_anatomical: torch.Tensor,
        z_temporal: torch.Tensor,
        z_systemic: torch.Tensor,
        recorder: EvidenceAttentionRecorder | None = None,
    ) -> EvidenceRepresentation:
        raise NotImplementedError


class StaticFusion(BaseEvidenceFusion):
    """Legacy static fusion mechanism bypassing attention."""

    def forward(
        self,
        z_symptom: torch.Tensor,
        z_anatomical: torch.Tensor,
        z_temporal: torch.Tensor,
        z_systemic: torch.Tensor,
        recorder: EvidenceAttentionRecorder | None = None,
    ) -> EvidenceRepresentation:
        if recorder is not None and recorder.record_enabled:
            trace = EvidenceReasoningTrace(fusion_type="StaticFusion")
            recorder.record(trace)

        return EvidenceRepresentation(
            symptom=z_symptom,
            anatomical=z_anatomical,
            temporal=z_temporal,
            systemic=z_systemic,
        )


class AttentionFusion(BaseEvidenceFusion):
    """Adaptive Clinical Evidence Synthesizer (ACES) fusion mechanism."""

    def __init__(self, config: EmergentPathTriageConfig) -> None:
        super().__init__()
        self.config = config
        self.latent_dim = config.latent_dim
        self.mode = config.aces_fusion_mode  # A1, A2, A3
        self.num_heads = config.aces_num_heads

        # 1. Aspect Identity (Clinical Aspect Prototypes)
        if self.mode == "A3":
            self.prototypes = nn.Parameter(torch.randn(4, self.latent_dim))
        else:
            self.register_parameter("prototypes", None)

        # 2. Aspect Interaction
        self.interaction = nn.MultiheadAttention(
            embed_dim=self.latent_dim,
            num_heads=self.num_heads,
            dropout=config.dces_dropout,
            batch_first=True,
        )

        # 3. Importance Predictor
        self.importance_predictor = nn.Sequential(
            nn.Linear(self.latent_dim, 1), nn.Sigmoid()
        )

        # 4. Residual and 5. Normalization
        if config.dces_normalization == "layernorm":
            self.norm = nn.LayerNorm(self.latent_dim)
        else:
            self.norm = nn.Identity()

    def forward(
        self,
        z_symptom: torch.Tensor,
        z_anatomical: torch.Tensor,
        z_temporal: torch.Tensor,
        z_systemic: torch.Tensor,
        recorder: EvidenceAttentionRecorder | None = None,
    ) -> EvidenceRepresentation:
        batch_size = z_symptom.shape[0]

        # Keys / Values: Raw aspect projections (Batch, 4, LatentDim)
        kv = torch.stack([z_symptom, z_anatomical, z_temporal, z_systemic], dim=1)

        # 1. Aspect Identity
        if self.mode == "A3":
            # Prototypes act as Queries
            q = self.prototypes.unsqueeze(0).expand(batch_size, 4, -1)
        else:
            q = kv

        # 2. Aspect Interaction
        refined_prototypes, interaction_weights = self.interaction(
            q, kv, kv, average_attn_weights=False
        )

        # 3. Importance Predictor
        importance_scores = self.importance_predictor(refined_prototypes).squeeze(-1)

        # 4. Gated Refinement
        gated = refined_prototypes * importance_scores.unsqueeze(-1)

        # 5. Residual Refinement
        if self.mode in ["A2", "A3"]:
            refined = kv + gated
        else:
            # A1: Attention Only
            refined = gated

        # 6. Normalization
        final_out = self.norm(refined)

        out_symptom, out_anatomical, out_temporal, out_systemic = final_out.unbind(
            dim=1
        )

        if recorder is not None and recorder.record_enabled:
            avg_interaction = interaction_weights.mean(dim=1)  # (Batch, 4, 4)
            log_weights = torch.log(avg_interaction + 1e-9)
            entropy = -torch.sum(avg_interaction * log_weights, dim=-1).mean().item()

            if self.mode == "A3":
                utilization = {}
                mean_refined = refined_prototypes.mean(dim=0)
                dists = torch.norm(mean_refined - self.prototypes, dim=1)
                keys = ["symptom", "anatomical", "temporal", "systemic"]
                for i, k in enumerate(keys):
                    utilization[k] = dists[i].item()
            else:
                utilization = None

            trace = EvidenceReasoningTrace(
                fusion_type="AttentionFusion",
                aspect_importance_scores={
                    "symptom": importance_scores[:, 0],
                    "anatomical": importance_scores[:, 1],
                    "temporal": importance_scores[:, 2],
                    "systemic": importance_scores[:, 3],
                },
                interaction_weights=interaction_weights,
                fusion_entropy=entropy,
                prototype_utilization=utilization,
            )
            recorder.record(trace)

        return EvidenceRepresentation(
            symptom=out_symptom,
            anatomical=out_anatomical,
            temporal=out_temporal,
            systemic=out_systemic,
        )


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
      3. Fusion: The projected aspects are fed into a BaseEvidenceFusion module.
    """

    def __init__(self, hidden_dim: int, config: EmergentPathTriageConfig) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.config = config
        self.recorder = EvidenceAttentionRecorder()

        # Modular token aggregator abstraction
        self.pooler = MaskedMeanPooler()

        # Reusable Projection Block Factory providing parameter isolation
        self.symptom_proj = ProjectionBlock(hidden_dim, config.latent_dim, config)
        self.anatomical_proj = ProjectionBlock(hidden_dim, config.latent_dim, config)
        self.temporal_proj = ProjectionBlock(hidden_dim, config.latent_dim, config)
        self.systemic_proj = ProjectionBlock(hidden_dim, config.latent_dim, config)

        # Fusion Interface Dependency
        if getattr(config, "aces_fusion_mode", "A0") == "A0":
            self.fusion = StaticFusion()
        else:
            self.fusion = AttentionFusion(config)

        logger.info(
            f"Initialized ClinicalEvidenceSynthesizer with hidden_dim={hidden_dim}, "
            f"latent_dim={config.latent_dim}, activation='{config.dces_activation}', "
            f"fusion_mode='{getattr(config, 'aces_fusion_mode', 'A0')}'"
        )

    def load_state_dict(
        self, state_dict: dict, strict: bool = True, assign: bool = False
    ):
        """Intercept load_state_dict to implement fallback to StaticFusion."""
        # Determine if the state dict contains any keys for the 'fusion' module
        # It's an older checkpoint if there are no 'fusion' keys
        has_fusion_keys = any("fusion." in k for k in state_dict)

        if not has_fusion_keys and not strict:
            if isinstance(self.fusion, AttentionFusion):
                logger.warning(
                    "Missing ACES fusion parameters in checkpoint. "
                    "Falling back to StaticFusion (A0 mode) to maintain backward compatibility."
                )
                self.fusion = StaticFusion()
                self.config.aces_fusion_mode = "A0"

        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def forward(
        self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> EvidenceRepresentation:
        """Synthesize orthogonal clinical aspect representations from transformer embeddings."""
        # 1. Device and dtype validations
        device = next(self.parameters()).device

        if not isinstance(token_embeddings, torch.Tensor):
            raise InterfaceError(
                f"token_embeddings must be a torch.Tensor, got {type(token_embeddings)}"
            )
        if not isinstance(attention_mask, torch.Tensor):
            raise InterfaceError(
                f"attention_mask must be a torch.Tensor, got {type(attention_mask)}"
            )

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
            raise InterfaceError(
                f"Incorrect dtype: token_embeddings must be torch.float32, got {token_embeddings.dtype}."
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
            raise InterfaceError("Batch dimension mismatch")
        if seq_len != mask_len:
            raise InterfaceError("Sequence dimension mismatch")
        if hidden_dim != self.hidden_dim:
            raise InterfaceError("Hidden dimension mismatch")

        # 3. Dynamic sequence-level aggregation
        pooled = self.pooler(token_embeddings, attention_mask)

        # 4. Aspect projections mapping
        if getattr(self.config, "ablation_dces_enabled", True):
            z_symptom = self.symptom_proj(pooled)
            z_anatomical = self.anatomical_proj(pooled)
            z_temporal = self.temporal_proj(pooled)
            z_systemic = self.systemic_proj(pooled)
        else:
            z_aspect = self.symptom_proj(pooled)
            z_symptom = z_aspect
            z_anatomical = z_aspect
            z_temporal = z_aspect
            z_systemic = z_aspect

        # Enforce clean zero output for fully padded sequences
        counts = attention_mask.sum(dim=1, keepdim=True)
        z_symptom = torch.where(counts == 0, torch.zeros_like(z_symptom), z_symptom)
        z_anatomical = torch.where(
            counts == 0, torch.zeros_like(z_anatomical), z_anatomical
        )
        z_temporal = torch.where(counts == 0, torch.zeros_like(z_temporal), z_temporal)
        z_systemic = torch.where(counts == 0, torch.zeros_like(z_systemic), z_systemic)

        # 5. Evidence Fusion
        return self.fusion(
            z_symptom, z_anatomical, z_temporal, z_systemic, recorder=self.recorder
        )
