"""Model registration, container wrappers, and checkpoint verifiers for E-PATH-CO-REASON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar
import torch
import torch.nn as nn
from transformers import AutoModel, XLMRobertaModel

from models.base_model import BaseMediTriageModel, load_tokenizer_or_fallback
from models.emergent_path_triage.config import EmergentPathTriageConfig
from models.emergent_path_triage.exceptions import CompatibilityError, InterfaceError
from models.emergent_path_triage.interfaces import BaseEmergentPathTriage, BaseCheckpointRegistry
from models.emergent_path_triage.logger import get_logger
from models.emergent_path_triage.types import (
    ModelOutputs,
    RoutingDecision,
    ThoughtPath,
)

logger = get_logger()


class EmergentPathTriageModel(BaseMediTriageModel):
    """E-PATH-CO-REASON entry in the MediTriage model zoo.
    
    Exposes immutable metadata and handles instantiation.
    """
    model_name: ClassVar[str] = "xlm-roberta-base"
    display_name: ClassVar[str] = "E-PATH-CO-REASON (Emergent Path Triage)"
    short_name: ClassVar[str] = "emergent_path_triage"
    is_novel_contribution: ClassVar[bool] = True

    # Immutable Research Metadata
    architecture_name: ClassVar[str] = "E-PATH-CO-REASON"
    architecture_version: ClassVar[str] = "1.0.0"
    paper_name: ClassVar[str] = "E-PATH-CO-REASON: Emergent Path-Aligned Co-evolutionary Triage Network"
    paper_version: ClassVar[str] = "1.0.0"
    research_stage: ClassVar[str] = "Phase 1 - Foundational Soft Skeletal Structure"
    model_family: ClassVar[str] = "Co-evolutionary Reasoning"

    @classmethod
    def build_tokenizer(cls):
        logger.info(f"Building tokenizer for E-PATH-CO-REASON using model name '{cls.model_name}'")
        return load_tokenizer_or_fallback(cls.model_name)

    @classmethod
    def build_encoder(cls, config: Any | None = None) -> XLMRobertaModel:
        logger.info(f"Building encoder for E-PATH-CO-REASON using model name '{cls.model_name}'")
        return AutoModel.from_pretrained(cls.model_name)

    def build(self, config: Any, triage_config: EmergentPathTriageConfig | None = None) -> nn.Module:
        """Build and wrap the emergent path model."""
        encoder = self.build_encoder(config)
        triage_cfg = triage_config or (
            EmergentPathTriageConfig.from_dict(config)
            if isinstance(config, dict)
            else (config if isinstance(config, EmergentPathTriageConfig) else EmergentPathTriageConfig())
        )
        return EmergentPathTriageTransformer(encoder, triage_cfg)


class EmergentPathTriageTransformer(BaseEmergentPathTriage):
    """The PyTorch Module implementing E-PATH-CO-REASON core workflow.

    ============================================================================
    DTYPE CONTRACT SPECIFICATION
    ============================================================================
    Module               | Input Dtype           | Output Dtype          | Precision Mode
    ---------------------+-----------------------+-----------------------+------------------
    Transformer Encoder  | torch.long (input_ids)| torch.float16/float32 | AMP Autocast (Safe)
    Evidence Synthesizer | torch.float32         | torch.float32         | float32 ONLY (Unsafe)
    Thought Blocks       | torch.float32         | torch.float32         | float32 ONLY (Unsafe)
    Router               | torch.float32         | torch.float32         | float32 ONLY (Unsafe)
    Execution Engine     | torch.float32         | torch.float32         | float32 ONLY (Unsafe)
    Consistency Proj.    | torch.float32         | torch.float32         | float32 ONLY (Unsafe)
    Prediction Heads     | torch.float32         | torch.float16/float32 | AMP Autocast (Safe)
    Loss Computation     | torch.float32         | torch.float32         | float32 ONLY
    ============================================================================
    All modules must run on the active device matching module parameters.
    """
    
    def __init__(self, encoder: XLMRobertaModel, config: EmergentPathTriageConfig) -> None:
        super().__init__()
        self.encoder = encoder
        self.config = config
        self.dropout = nn.Dropout(0.1)
        
        # Keep classifiers as parameters for optimizer setup compatibility
        hidden_size = encoder.config.hidden_size
        
        # Instantiate the two independent Prediction Heads mapping the final reasoning state
        from models.emergent_path_triage.heads import PredictionHead
        self.classifier_specialist = PredictionHead(config.latent_dim, 13, config)
        self.classifier_severity = PredictionHead(config.latent_dim, 5, config)
        
        # Instantiate the Dynamic Clinical Evidence Synthesizer (DCES)
        from models.emergent_path_triage.dces import ClinicalEvidenceSynthesizer
        self.dces = ClinicalEvidenceSynthesizer(hidden_size, config)
        
        # Instantiate the Dynamic Clinical Reasoning Router (DCRR)
        from models.emergent_path_triage.dcrr import ClinicalReasoningRouter
        self.router = ClinicalReasoningRouter(config)
        
        # Instantiate the Emergent Clinical Thought Blocks (CTBs)
        from models.emergent_path_triage.ctb import ClinicalThoughtBlock
        self.blocks = nn.ModuleList([
            ClinicalThoughtBlock(config.latent_dim, config)
            for _ in range(config.num_thought_blocks)
        ])
        
        # Instantiate the Reasoning Path Execution Engine
        from models.emergent_path_triage.engine import ReasoningPathExecutionEngine
        self.engine = ReasoningPathExecutionEngine(config)
        
        # Instantiate the Dynamic Consistency Projection (DCP)
        from models.emergent_path_triage.dcp import DynamicConsistencyProjection
        self.dcp = DynamicConsistencyProjection(config.latent_dim, config)
        
        self.routing_seed: int = 42
        self.initialize_weights()

    def get_input_embeddings(self) -> nn.Module:
        return self.encoder.get_input_embeddings()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding | None:
        logger.info(f"Resizing token embeddings to: {new_num_tokens}")
        return self.encoder.resize_token_embeddings(new_num_tokens)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> ModelOutputs:
        """Process inputs and return strongly typed outputs."""
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Verify device mapping compliance
        self._verify_device_compliance(input_ids)
        self._verify_device_compliance(attention_mask)
        
        # Extract token level representations from the encoder (under outer AMP context)
        encoder_output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = encoder_output.last_hidden_state
        
        # ----------------------------------------------------------------------
        # SINGLE FLOAT32 ENTRY BOUNDARY & SELECTIVE autocast BOUNDARIES
        # ----------------------------------------------------------------------
        # The reasoning engine, routing, evidence synthesizer and thought blocks
        # must execute entirely in float32. We disable autocast selectively.
        device_type = "cuda" if device.type == "cuda" else "cpu"
        with torch.amp.autocast(device_type=device_type, enabled=False):
            # One centralized float32 restoration immediately before entering reasoning pipeline
            token_embeddings_f32 = token_embeddings.float()
            
            # Synthesize clinical evidence using DCES
            evidence = self.dces(token_embeddings_f32, attention_mask)
            
            # Determine reasoning paths using DCRR (Gumbel-Softmax)
            routing_decision = self.router(evidence, self.config.temperature)
            
            # Execute reasoning path through selected Clinical Thought Blocks
            evidence_list = [evidence.symptom, evidence.anatomical, evidence.temporal, evidence.systemic]
            final_state, thought_path = self.engine(evidence_list, routing_decision, self.blocks)
            
            # Centralized float32 assertion for prediction heads input
            final_state_f32 = final_state.float()
            
        # Map through prediction heads (re-entering safe outer autocast context if active)
        specialist_logits = self.classifier_specialist(final_state_f32)
        severity_logits = self.classifier_severity(final_state_f32)
        
        # Save intermediate states for modular loss hooks
        self._last_evidence = evidence
        self._last_routing_decision = routing_decision
        self._last_final_state = final_state_f32
        
        return ModelOutputs(
            specialist_logits=specialist_logits,
            severity_logits=severity_logits,
            routing_decision=routing_decision,
            thought_path=thought_path
        )

    def compute_loss(
        self,
        specialist_logits: torch.Tensor,
        severity_logits: torch.Tensor,
        labels_specialist: torch.Tensor,
        labels_severity: torch.Tensor,
        joint_loss_fn: nn.Module,
    ) -> dict[str, torch.Tensor]:
        """Compute task-consistency loss and regularized objectives.
        
        This architecture-agnostic API encapsulates base loss calculations
        and merges E-PATH-CO-REASON auxiliary objectives.
        """
        # Enforce float32 dtype contract for all loss computations
        specialist_logits = specialist_logits.float()
        severity_logits = severity_logits.float()
        device = specialist_logits.device
        
        # Verify shape requirements
        if len(specialist_logits.shape) != 2 or specialist_logits.shape[1] != 13:
            raise InterfaceError(f"Incorrect specialist_logits shape: {specialist_logits.shape}")
        if len(severity_logits.shape) != 2 or severity_logits.shape[1] != 5:
            raise InterfaceError(f"Incorrect severity_logits shape: {severity_logits.shape}")
            
        # Calculate standard classification loss elements
        base_losses = joint_loss_fn(specialist_logits, severity_logits, labels_specialist, labels_severity)
        
        # 1. Orthogonality Regularization Loss
        if self.config.ortho_lambda > 0.0 and hasattr(self, "_last_evidence") and self._last_evidence is not None:
            similarities = self._last_evidence.compute_pairwise_similarities()
            identity = torch.eye(4, device=similarities.device).unsqueeze(0)
            ortho_loss = torch.mean((similarities - identity) ** 2)
        else:
            ortho_loss = torch.zeros((), device=device)
            
        # 2. Dynamic Consistency Projection (DCP) Loss
        if self.config.cons_lambda > 0.0 and hasattr(self, "_last_final_state") and self._last_final_state is not None:
            # Concatenate specialist and severity logits: (Batch, 18)
            preds = torch.cat([specialist_logits, severity_logits], dim=-1)
            # Project through DCP
            proj_reasoning, proj_preds = self.dcp(self._last_final_state, preds)
            # Alignment error (MSE loss)
            cons_loss = torch.mean((proj_reasoning - proj_preds) ** 2)
        else:
            cons_loss = torch.zeros((), device=device)
            
        # 3. DCRR Routing Diversity Loss
        if self.config.div_lambda > 0.0 and hasattr(self, "_last_routing_decision") and self._last_routing_decision is not None:
            routing_probs = self._last_routing_decision.routing_probabilities
            # Mean probability of each block across batch and steps: (Num_Blocks,)
            mean_probs = routing_probs.mean(dim=(0, 1))
            div_loss = torch.sum(mean_probs * torch.log(mean_probs + 1e-10))
        else:
            div_loss = torch.zeros((), device=device)
            
        # Accumulate total augmented loss
        total_loss = (
            base_losses["joint_loss"]
            + self.config.ortho_lambda * ortho_loss
            + self.config.cons_lambda * cons_loss
            + self.config.div_lambda * div_loss
        )
        
        return {
            "joint_loss": total_loss,
            "specialist_loss": base_losses["specialist_loss"],
            "severity_loss": base_losses["severity_loss"],
            "ortho_loss": ortho_loss.detach(),
            "cons_loss": cons_loss.detach(),
            "div_loss": div_loss.detach(),
        }

    def initialize_weights(self) -> None:
        """Deterministic initialization of auxiliary heads."""
        logger.info(f"Initializing E-PATH-CO-REASON head weights with seed={self.routing_seed}")
        torch.manual_seed(self.routing_seed)
        
        # Initialize classifier_specialist layers
        nn.init.xavier_uniform_(self.classifier_specialist.fc1.weight)
        if self.classifier_specialist.fc1.bias is not None:
            nn.init.zeros_(self.classifier_specialist.fc1.bias)
        nn.init.xavier_uniform_(self.classifier_specialist.fc2.weight)
        if self.classifier_specialist.fc2.bias is not None:
            nn.init.zeros_(self.classifier_specialist.fc2.bias)
        
        # Initialize classifier_severity layers
        nn.init.xavier_uniform_(self.classifier_severity.fc1.weight)
        if self.classifier_severity.fc1.bias is not None:
            nn.init.zeros_(self.classifier_severity.fc1.bias)
        nn.init.xavier_uniform_(self.classifier_severity.fc2.weight)
        if self.classifier_severity.fc2.bias is not None:
            nn.init.zeros_(self.classifier_severity.fc2.bias)
            
        # Initialize DynamicConsistencyProjection layers
        nn.init.xavier_uniform_(self.dcp.reasoning_proj.weight)
        nn.init.xavier_uniform_(self.dcp.logits_proj.weight)

    def reset_parameters(self) -> None:
        """Reset parameter weights to defaults."""
        self.initialize_weights()

    def set_seed(self, seed: int) -> None:
        """Set routing random seed."""
        self.routing_seed = seed
        self.initialize_weights()


class EmergentPathCheckpointRegistry(BaseCheckpointRegistry):
    """Implementation of metadata serialization and verification contracts."""

    def save_checkpoint_metadata(self, path: Path, metadata: dict[str, Any]) -> None:
        """Persist checkpoint compatibility metadata file."""
        meta_file = path / "checkpoint_metadata.json" if path.is_dir() else path
        logger.info(f"Saving checkpoint metadata to: {meta_file}")
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4, sort_keys=True)

    def load_checkpoint_metadata(self, path: Path) -> dict[str, Any]:
        """Load checkpoint compatibility metadata."""
        meta_file = path / "checkpoint_metadata.json" if path.is_dir() else path
        logger.info(f"Loading checkpoint metadata from: {meta_file}")
        if not meta_file.exists():
            raise CompatibilityError(f"Checkpoint metadata file not found at: {meta_file}")
        with open(meta_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def verify_compatibility(self, checkpoint_meta: dict[str, Any], current_config: EmergentPathTriageConfig) -> bool:
        """Verify if a checkpoint file matches running architecture requirements."""
        input_ver = checkpoint_meta.get("compatibility_version")
        current_ver = current_config.compatibility_version
        if input_ver != current_ver:
            raise CompatibilityError(
                f"Checkpoint compatibility version '{input_ver}' does not match "
                f"running architecture compatibility version '{current_ver}'."
            )
            
        # Verify layer dimension alignments
        checkpoint_dim = checkpoint_meta.get("latent_dim")
        if checkpoint_dim != current_config.latent_dim:
            raise CompatibilityError(
                f"Checkpoint latent dimension {checkpoint_dim} does not match "
                f"running configuration latent dimension {current_config.latent_dim}."
            )
        
        logger.info("Checkpoint compatibility verification passed.")
        return True
