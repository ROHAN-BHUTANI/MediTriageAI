"""Abstract interfaces and explicit contracts for E-PATH-CO-REASON.

These classes define the API, shapes, dtypes, and device expectations for all 
future neural components. All interfaces explicitly document stable public methods,
protected helpers, and extension points.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
import torch
import torch.nn as nn

from models.emergent_path_triage.exceptions import InterfaceError, CompatibilityError
from models.emergent_path_triage.config import EmergentPathTriageConfig
from models.emergent_path_triage.types import (
    EvidenceRepresentation,
    RoutingDecision,
    ThoughtPath,
    ModelOutputs,
)


class BaseClinicalEvidenceSynthesizer(nn.Module, ABC):
    """Abstract interface for the Dynamic Clinical Evidence Synthesizer (DCES).
    
    API STABILITY:
    - Stable Public Interface: forward() method.
    - Extension Points: Subclasses must override forward().
    """
    
    @abstractmethod
    def forward(
        self, 
        token_embeddings: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> EvidenceRepresentation:
        """Extract evidence representations.
        
        Args:
            token_embeddings: Tensor containing token-level context states.
                Shape: (Batch_Size, Sequence_Length, Hidden_Dimension)
                Dtype: torch.float32
                Device: Match module parameters device.
            attention_mask: Mask denoting valid tokens.
                Shape: (Batch_Size, Sequence_Length)
                Dtype: torch.long or torch.bool
                Device: Match module parameters device.
                
        Returns:
            An EvidenceRepresentation instance where each aspect tensor has:
                Shape: (Batch_Size, Latent_Dimension)
                Dtype: torch.float32
        """
        raise NotImplementedError


class BaseReasoningRouter(nn.Module, ABC):
    """Abstract interface for the Dynamic Clinical Reasoning Router (DCRR).
    
    API STABILITY:
    - Stable Public Interface: forward() method.
    - Extension Points: Subclasses must override forward().
    """
    
    @abstractmethod
    def forward(
        self, 
        evidence: EvidenceRepresentation, 
        temperature: float
    ) -> RoutingDecision:
        """Compute the dynamic routing decision.
        
        Args:
            evidence: Evidence representations to route.
                Dtype: torch.float32
            temperature: Gumbel-Softmax scaling factor.
                Dtype: float (scalar)
                
        Returns:
            A RoutingDecision object. The probabilities tensor has:
                Shape: (Batch_Size, Max_Path_Depth, Num_Blocks)
                Dtype: torch.float32
        """
        raise NotImplementedError


class BaseClinicalThoughtBlock(nn.Module, ABC):
    """Abstract interface for a Clinical Thought Block (CTB).
    
    API STABILITY:
    - Stable Public Interface: forward() method.
    - Extension Points: Subclasses must override forward().
    """
    
    @abstractmethod
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Perform clinical thought update.
        
        Args:
            state: Contextual latent embedding representation.
                Shape: (Batch_Size, Latent_Dimension)
                Dtype: torch.float32
                Device: Match module parameters device.
                
        Returns:
            Updated latent embedding representation.
                Shape: (Batch_Size, Latent_Dimension)
                Dtype: torch.float32
        """
        raise NotImplementedError


class BaseConsistencyProjection(nn.Module, ABC):
    """Abstract interface for the Dynamic Consistency Projection (DCP).
    
    API STABILITY:
    - Stable Public Interface: forward() method.
    - Extension Points: Subclasses must override forward().
    """
    
    @abstractmethod
    def forward(
        self, 
        specialist_state: torch.Tensor, 
        severity_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project specialist and severity states to the urgency space.
        
        Args:
            specialist_state: Specialist pathway representation.
                Shape: (Batch_Size, Hidden_Dimension)
                Dtype: torch.float32
                Device: Match module parameters device.
            severity_state: Severity pathway representation.
                Shape: (Batch_Size, Hidden_Dimension)
                Dtype: torch.float32
                Device: Match module parameters device.
                
        Returns:
            A tuple of (specialist_urgency, severity_urgency) projected tensors.
                Each projected tensor has:
                    Shape: (Batch_Size, Urgency_Dimension)
                    Dtype: torch.float32
        """
        raise NotImplementedError


class BaseEmergentPathTriage(nn.Module, ABC):
    """Abstract interface for the top-level E-PATH-CO-REASON container.
    
    API STABILITY:
    - Stable Public Interface: forward(), compute_loss(), set_seed(), reset_parameters()
    - Protected Hooks: _verify_device_compliance()
    - Extension Points: Weight initializers and custom forward components.
    """
    
    @abstractmethod
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> ModelOutputs:
        """Perform the full triage forward pass.
        
        Args:
            input_ids: Token indices representing clinical complaint.
                Shape: (Batch_Size, Sequence_Length)
                Dtype: torch.long
                Device: Match model device.
            attention_mask: Mask denoting valid input tokens.
                Shape: (Batch_Size, Sequence_Length)
                Dtype: torch.long or torch.bool
                Device: Match model device.
                
        Returns:
            A ModelOutputs object bundling predictions and routing decisions.
        """
        raise NotImplementedError

    @abstractmethod
    def compute_loss(
        self,
        specialist_logits: torch.Tensor,
        severity_logits: torch.Tensor,
        labels_specialist: torch.Tensor,
        labels_severity: torch.Tensor,
        joint_loss_fn: nn.Module,
    ) -> dict[str, torch.Tensor]:
        """Compute task-consistency loss and regularized objectives.
        
        Args:
            specialist_logits: Output logits for specialists: (Batch_Size, 13)
            severity_logits: Output logits for severity: (Batch_Size, 5)
            labels_specialist: Ground truth specialist IDs: (Batch_Size,)
            labels_severity: Ground truth severity IDs: (Batch_Size,)
            joint_loss_fn: Standard JointLoss module.
            
        Returns:
            A dictionary containing:
                "joint_loss": Total regularized loss scalar tensor.
                "specialist_loss": Cross-entropy specialist scalar tensor.
                "severity_loss": Cross-entropy severity scalar tensor.
        """
        raise NotImplementedError

    @abstractmethod
    def initialize_weights(self) -> None:
        """Initialize all neural layer weights deterministically."""
        raise NotImplementedError

    @abstractmethod
    def reset_parameters(self) -> None:
        """Reset parameter weights to baseline states."""
        raise NotImplementedError

    @abstractmethod
    def set_seed(self, seed: int) -> None:
        """Set seed values for internal routing mechanisms."""
        raise NotImplementedError

    def _verify_device_compliance(self, tensor: torch.Tensor) -> None:
        """Protected utility helper to verify device mapping alignment."""
        device = next(self.parameters()).device
        if tensor.device != device:
            raise InterfaceError(f"Tensor device mismatch: expected {device}, got {tensor.device}")


class BaseCheckpointRegistry(ABC):
    """Abstract interface defining the metadata contract for checkpoint compatibility."""

    @abstractmethod
    def save_checkpoint_metadata(self, path: Path, metadata: dict[str, Any]) -> None:
        """Persist checkpoint compatibility metadata file.
        
        Args:
            path: Destination directory or file path.
            metadata: Compatibility attributes (schema versions, weights structure).
        """
        raise NotImplementedError

    @abstractmethod
    def load_checkpoint_metadata(self, path: Path) -> dict[str, Any]:
        """Load checkpoint compatibility metadata.
        
        Args:
            path: Source directory or metadata file path.
            
        Returns:
            Dictionary containing schema version and compatibility metadata.
        """
        raise NotImplementedError

    @abstractmethod
    def verify_compatibility(self, checkpoint_meta: dict[str, Any], current_config: EmergentPathTriageConfig) -> bool:
        """Verify if a checkpoint file is compatible with the current architecture configuration.
        
        Args:
            checkpoint_meta: Deserialized checkpoint metadata.
            current_config: Running model configuration instance.
            
        Returns:
            True if compatible, raises CompatibilityError otherwise.
        """
        raise NotImplementedError
