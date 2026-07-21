"""Strongly typed dataclasses with self-validation and serialization for E-PATH-CO-REASON."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import torch

from models.emergent_path_triage.exceptions import InterfaceError, RoutingError


def _to_list_if_tensor(val: Any) -> Any:
    if isinstance(val, torch.Tensor):
        return val.detach().cpu().tolist()
    return val


def _to_tensor_if_list(val: Any, device: str = "cpu") -> Any:
    if isinstance(val, list):
        return torch.tensor(val, dtype=torch.float32, device=torch.device(device))
    if isinstance(val, (int, float)):
        return torch.tensor(val, dtype=torch.float32, device=torch.device(device))
    return val



@dataclass(frozen=True)
class EvidenceRepresentation:
    """Four-aspect latent clinical evidence representations.
    
    Tensors represent projection states for each aspect of symptom history.
    Typical shape: (Batch, Latent_Dim)
    Dtype: torch.float32
    """
    symptom: torch.Tensor
    anatomical: torch.Tensor
    temporal: torch.Tensor
    systemic: torch.Tensor

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate tensor shapes and metadata."""
        aspects = {"symptom": self.symptom, "anatomical": self.anatomical, "temporal": self.temporal, "systemic": self.systemic}
        batch_size = None
        for name, tensor in aspects.items():
            if not isinstance(tensor, torch.Tensor):
                raise InterfaceError(f"Evidence {name} must be a torch.Tensor, got {type(tensor)}")
            if len(tensor.shape) != 2:
                raise InterfaceError(f"Evidence {name} must be a 2D tensor of shape (Batch, Latent_Dim), got shape {tensor.shape}")
            if batch_size is None:
                batch_size = tensor.shape[0]
            elif tensor.shape[0] != batch_size:
                raise InterfaceError(f"Batch size mismatch in Evidence: {name} has batch size {tensor.shape[0]} but symptom has {batch_size}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize evidence representations into a deterministic dictionary of lists."""
        return {
            "symptom": _to_list_if_tensor(self.symptom),
            "anatomical": _to_list_if_tensor(self.anatomical),
            "temporal": _to_list_if_tensor(self.temporal),
            "systemic": _to_list_if_tensor(self.systemic),
        }

    def compute_pairwise_similarities(self) -> torch.Tensor:
        """Compute pairwise cosine similarities among the four clinical aspects.
        
        Returns:
            Cosine similarity matrix of shape (Batch_Size, 4, 4).
            Indices: 0: Symptom, 1: Anatomical, 2: Temporal, 3: Systemic.
        """
        stacked = torch.stack([self.symptom, self.anatomical, self.temporal, self.systemic], dim=1)
        norms = torch.norm(stacked, p=2, dim=2, keepdim=True).clamp(min=1e-8)
        normalized = stacked / norms
        # Pairwise dot products via batch matrix multiplication (Batch, 4, d) x (Batch, d, 4) -> (Batch, 4, 4)
        return torch.bmm(normalized, normalized.transpose(1, 2))


    @classmethod
    def from_dict(cls, data: dict[str, Any], device: str = "cpu") -> EvidenceRepresentation:
        """De-serialize dictionary of lists into PyTorch evidence representations."""
        return cls(
            symptom=_to_tensor_if_list(data["symptom"], device),
            anatomical=_to_tensor_if_list(data["anatomical"], device),
            temporal=_to_tensor_if_list(data["temporal"], device),
            systemic=_to_tensor_if_list(data["systemic"], device),
        )


@dataclass(frozen=True)
class RoutingDecision:
    """Strongly typed outputs from the Gumbel-Softmax Router (DCRR).
    
    Includes routing probability matrices and auditing identifiers.
    """
    routing_logits: torch.Tensor         # Shape: (Batch, Path_Depth, Num_Blocks)
    routing_probabilities: torch.Tensor  # Shape: (Batch, Path_Depth, Num_Blocks)
    selected_blocks: list[int]          # List of indices of chosen thought blocks
    path_depth: int                      # Actual depth traversed
    routing_entropy: torch.Tensor        # Entropy penalty scalar: (1,)
    routing_confidence: torch.Tensor     # Confidence probability scalar: (1,)
    path_identifier: str                 # Unique hash identifier for interpretability auditing

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate routing decision bounds and shapes."""
        if not isinstance(self.routing_logits, torch.Tensor):
            raise RoutingError("routing_logits must be a torch.Tensor")
        if not isinstance(self.routing_probabilities, torch.Tensor):
            raise RoutingError("routing_probabilities must be a torch.Tensor")
            
        if len(self.routing_logits.shape) != 3:
            raise RoutingError(
                f"routing_logits must be 3D of shape (Batch, Depth, Blocks), got {self.routing_logits.shape}"
            )
        if len(self.routing_probabilities.shape) != 3:
            raise RoutingError(
                f"routing_probabilities must be 3D of shape (Batch, Depth, Blocks), got {self.routing_probabilities.shape}"
            )
            
        if self.routing_logits.shape != self.routing_probabilities.shape:
            raise RoutingError(
                f"Shape mismatch: routing_logits {self.routing_logits.shape} "
                f"does not match routing_probabilities {self.routing_probabilities.shape}"
            )
            
        if self.path_depth <= 0:
            raise RoutingError(f"path_depth must be strictly positive, got {self.path_depth}")
        if not self.selected_blocks:
            raise RoutingError("selected_blocks path cannot be empty")
        
        # Verify scalar losses/metrics
        for name, tensor in {"routing_entropy": self.routing_entropy, "routing_confidence": self.routing_confidence}.items():
            if not isinstance(tensor, torch.Tensor):
                raise RoutingError(f"{name} must be a torch.Tensor")
            if tensor.numel() != 1:
                raise RoutingError(f"{name} must be a scalar, got shape {tensor.shape}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize routing decisions to a deterministic dictionary."""
        return {
            "routing_logits": _to_list_if_tensor(self.routing_logits),
            "routing_probabilities": _to_list_if_tensor(self.routing_probabilities),
            "selected_blocks": list(self.selected_blocks),
            "path_depth": self.path_depth,
            "routing_entropy": _to_list_if_tensor(self.routing_entropy),
            "routing_confidence": _to_list_if_tensor(self.routing_confidence),
            "path_identifier": self.path_identifier,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], device: str = "cpu") -> RoutingDecision:
        """Construct RoutingDecision from dictionary."""
        return cls(
            routing_logits=_to_tensor_if_list(data["routing_logits"], device),
            routing_probabilities=_to_tensor_if_list(data["routing_probabilities"], device),
            selected_blocks=data["selected_blocks"],
            path_depth=data["path_depth"],
            routing_entropy=_to_tensor_if_list(data["routing_entropy"], device),
            routing_confidence=_to_tensor_if_list(data["routing_confidence"], device),
            path_identifier=data["path_identifier"],
        )


@dataclass(frozen=True)
class ThoughtPath:
    """Sequence of active thought blocks and intermediate states."""
    states: list[int]                    # List of block indices traversed
    representations: list[torch.Tensor]  # Latent embeddings at each step: list of (Batch, Latent_Dim)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate state counts and embedding shapes."""
        if not self.states:
            raise RoutingError("ThoughtPath states sequence cannot be empty")
        for i, t in enumerate(self.representations):
            if not isinstance(t, torch.Tensor):
                raise RoutingError(f"ThoughtPath representation at step {i} must be a torch.Tensor")
            if len(t.shape) != 2:
                raise RoutingError(f"ThoughtPath representation at step {i} must be 2D, got shape {t.shape}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize ThoughtPath to dictionary."""
        return {
            "states": list(self.states),
            "representations": [_to_list_if_tensor(r) for r in self.representations],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], device: str = "cpu") -> ThoughtPath:
        """Construct ThoughtPath from dictionary."""
        return cls(
            states=data["states"],
            representations=[_to_tensor_if_list(r, device) for r in data["representations"]],
        )


@dataclass(frozen=True)
class ModelOutputs:
    """Primary output wrapper for the emergent path triage model."""
    specialist_logits: torch.Tensor      # Shape: (Batch, Num_Specialists)
    severity_logits: torch.Tensor        # Shape: (Batch, Num_Severity_Labels)
    routing_decision: RoutingDecision | None = None
    thought_path: ThoughtPath | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate output logit dimensions."""
        if not isinstance(self.specialist_logits, torch.Tensor):
            raise InterfaceError("specialist_logits must be a torch.Tensor")
        if not isinstance(self.severity_logits, torch.Tensor):
            raise InterfaceError("severity_logits must be a torch.Tensor")
        
        if len(self.specialist_logits.shape) != 2 or self.specialist_logits.shape[1] != 13:
            raise InterfaceError(
                f"specialist_logits must have shape (Batch, 13), got {self.specialist_logits.shape}"
            )
        if len(self.severity_logits.shape) != 2 or self.severity_logits.shape[1] != 5:
            raise InterfaceError(
                f"severity_logits must have shape (Batch, 5), got {self.severity_logits.shape}"
            )

    def __iter__(self):
        return iter((self.specialist_logits, self.severity_logits))

    def to_dict(self) -> dict[str, Any]:
        """Serialize ModelOutputs to dictionary."""
        return {
            "specialist_logits": _to_list_if_tensor(self.specialist_logits),
            "severity_logits": _to_list_if_tensor(self.severity_logits),
            "routing_decision": self.routing_decision.to_dict() if self.routing_decision else None,
            "thought_path": self.thought_path.to_dict() if self.thought_path else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], device: str = "cpu") -> ModelOutputs:
        """Construct ModelOutputs from dictionary."""
        routing_dec = data.get("routing_decision")
        thought = data.get("thought_path")
        return cls(
            specialist_logits=_to_tensor_if_list(data["specialist_logits"], device),
            severity_logits=_to_tensor_if_list(data["severity_logits"], device),
            routing_decision=RoutingDecision.from_dict(routing_dec, device) if routing_dec else None,
            thought_path=ThoughtPath.from_dict(thought, device) if thought else None,
        )


@dataclass(frozen=True)
class AuxiliaryLosses:
    """Bundles the three auxiliary regularizers for co-evolutionary calibration."""
    ortho_loss: torch.Tensor             # Cosine orthogonality penalty scalar
    cons_loss: torch.Tensor              # Urgency manifold alignment loss scalar
    div_loss: torch.Tensor               # Router entropy maximization loss scalar

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate scalar properties of loss tensors."""
        losses = {"ortho_loss": self.ortho_loss, "cons_loss": self.cons_loss, "div_loss": self.div_loss}
        for name, tensor in losses.items():
            if not isinstance(tensor, torch.Tensor):
                raise InterfaceError(f"Loss {name} must be a torch.Tensor, got {type(tensor)}")
            if tensor.numel() != 1:
                raise InterfaceError(f"Loss {name} must be a scalar, got shape {tensor.shape}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize losses to dictionary."""
        return {
            "ortho_loss": _to_list_if_tensor(self.ortho_loss),
            "cons_loss": _to_list_if_tensor(self.cons_loss),
            "div_loss": _to_list_if_tensor(self.div_loss),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], device: str = "cpu") -> AuxiliaryLosses:
        """Construct AuxiliaryLosses from dictionary."""
        return cls(
            ortho_loss=_to_tensor_if_list(data["ortho_loss"], device),
            cons_loss=_to_tensor_if_list(data["cons_loss"], device),
            div_loss=_to_tensor_if_list(data["div_loss"], device),
        )
