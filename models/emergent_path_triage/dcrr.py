"""Dynamic Clinical Reasoning Router (DCRR) implementation for E-PATH-CO-REASON."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.emergent_path_triage.config import EmergentPathTriageConfig
from models.emergent_path_triage.exceptions import InterfaceError, RoutingError
from models.emergent_path_triage.interfaces import BaseReasoningRouter
from models.emergent_path_triage.logger import get_logger
from models.emergent_path_triage.types import EvidenceRepresentation, RoutingDecision

logger = get_logger()


class ClinicalReasoningRouter(BaseReasoningRouter):
    """Dynamic Clinical Reasoning Router (DCRR).
    
    Transforms the four latent aspect evidence projections from DCES into a 
    differentiable, Gumbel-Softmax-guided clinical reasoning path.
    """

    def __init__(self, config: EmergentPathTriageConfig) -> None:
        super().__init__()
        self.config = config

        # Validate setup variables
        if config.num_thought_blocks <= 0:
            raise RoutingError(f"num_thought_blocks must be positive, got {config.num_thought_blocks}")
        if config.max_path_depth <= 0:
            raise RoutingError(f"max_path_depth must be positive, got {config.max_path_depth}")
        if config.routing_hidden_dim <= 0:
            raise RoutingError(f"routing_hidden_dim must be positive, got {config.routing_hidden_dim}")

        # Independent, step-specific MLPs for each reasoning step
        # Preserves parameter independence and paths differentiation at each step.
        self.routing_steps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(4 * config.latent_dim, config.routing_hidden_dim),
                nn.GELU(),
                nn.Linear(config.routing_hidden_dim, config.num_thought_blocks)
            )
            for _ in range(config.max_path_depth)
        ])

        logger.info(
            f"Initialized ClinicalReasoningRouter with num_thought_blocks={config.num_thought_blocks}, "
            f"max_path_depth={config.max_path_depth}, routing_hidden_dim={config.routing_hidden_dim}"
        )

    def forward(
        self, 
        evidence: EvidenceRepresentation, 
        temperature: float
    ) -> RoutingDecision:
        """Compute the dynamic routing decision from aspect evidence.
        
        Complexity:
        - Time: O(B * M * (4 * d * H_r + H_r * N)) where H_r=routing_hidden_dim, N=num_blocks.
        - Space: O(B * M * N) memory allocation.
        """
        # 1. Input validations
        if not isinstance(evidence, EvidenceRepresentation):
            raise InterfaceError(
                f"Router input must be an EvidenceRepresentation dataclass, got {type(evidence)}"
            )
        if temperature <= 0.0:
            raise RoutingError(f"Gumbel-Softmax temperature must be strictly positive, got {temperature}")

        # Check devices
        device = next(self.parameters()).device
        for name, tensor in {
            "symptom": evidence.symptom,
            "anatomical": evidence.anatomical,
            "temporal": evidence.temporal,
            "systemic": evidence.systemic,
        }.items():
            if tensor.device != device:
                raise InterfaceError(
                    f"Device mismatch: Evidence aspect '{name}' resides on {tensor.device} "
                    f"but router parameters are on {device}"
                )

        batch_size = evidence.symptom.shape[0]

        # 2. Evidence Fusion stage (concatenate aspects to preserve identity)
        # Shape: (Batch_Size, 4 * Latent_Dim)
        fused = torch.cat(
            [evidence.symptom, evidence.anatomical, evidence.temporal, evidence.systemic],
            dim=-1
        )

        # 3. Predict step-specific routing logits
        step_logits_list = []
        for step_mlp in self.routing_steps:
            # Output shape: (Batch_Size, Num_Blocks)
            step_logits_list.append(step_mlp(fused))

        # Stack to shape: (Batch_Size, Max_Path_Depth, Num_Blocks)
        logits = torch.stack(step_logits_list, dim=1)

        # 4. Numerically stable Gumbel-Softmax routing execution
        probs = torch.softmax(logits, dim=-1)

        if self.training:
            # Draw Gumbel noise: g = -log(-log(u))
            u = torch.rand_like(logits)
            eps = 1e-10
            gumbel_noise = -torch.log(-torch.log(u + eps) + eps)
            
            # Apply soft routing probabilities (differentiable Gumbel-Softmax)
            routing_probs = torch.softmax((logits + gumbel_noise) / temperature, dim=-1)
        else:
            # Deterministic hard routing during inference (argmax selection mapped to one-hot)
            hard_indices = torch.argmax(logits, dim=-1)
            routing_probs = torch.zeros_like(logits).scatter_(-1, hard_indices.unsqueeze(-1), 1.0)

        # 5. Extract metadata outputs
        # Compute default hard selections for audit identifiers
        with torch.no_grad():
            clean_probs = torch.softmax(logits, dim=-1)
            hard_indices = torch.argmax(clean_probs, dim=-1)
            # Choose representative path for the first sample in batch
            selected_path = hard_indices[0].tolist()

        # Compute entropy penalty: H(P) = -sum(p * log(p)) averaged across batch and steps
        # Used for routing diversity loss regularizers
        entropy = -torch.sum(clean_probs * torch.log(clean_probs + 1e-10), dim=-1).mean()

        # Compute confidence: mean of max probabilities selected
        confidence = clean_probs.max(dim=-1)[0].mean()

        # Create unique path identifier audit code
        prefix = "train_soft_path_" if self.training else "infer_hard_path_"
        path_identifier = prefix + "-".join(map(str, selected_path))

        return RoutingDecision(
            routing_logits=logits,
            routing_probabilities=routing_probs,
            selected_blocks=selected_path,
            path_depth=self.config.max_path_depth,
            routing_entropy=entropy,
            routing_confidence=confidence,
            path_identifier=path_identifier
        )
