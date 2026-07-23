"""Strongly typed dataclasses with self-validation and serialization for E-PATH-CO-REASON."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
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
    routing_trace: RoutingTrace | None = None
    thought_path: ThoughtPath | None = None
    specialist_confidence: ClinicalConfidenceOutput | None = None
    severity_confidence: ClinicalConfidenceOutput | None = None

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
            "routing_trace": self.routing_trace.to_dict() if self.routing_trace else None,
            "thought_path": self.thought_path.to_dict() if self.thought_path else None,
            "specialist_confidence": self.specialist_confidence.to_dict() if self.specialist_confidence else None,
            "severity_confidence": self.severity_confidence.to_dict() if self.severity_confidence else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], device: str = "cpu") -> ModelOutputs:
        """Construct ModelOutputs from dictionary."""
        routing_dec = data.get("routing_decision")
        routing_tr = data.get("routing_trace")
        thought = data.get("thought_path")
        spec_conf = data.get("specialist_confidence")
        sev_conf = data.get("severity_confidence")
        return cls(
            specialist_logits=_to_tensor_if_list(data["specialist_logits"], device),
            severity_logits=_to_tensor_if_list(data["severity_logits"], device),
            routing_decision=RoutingDecision.from_dict(routing_dec, device) if routing_dec else None,
            routing_trace=RoutingTrace.from_dict(routing_tr, device) if routing_tr else None,
            thought_path=ThoughtPath.from_dict(thought, device) if thought else None,
            specialist_confidence=ClinicalConfidenceOutput.from_dict(spec_conf, device) if spec_conf else None,
            severity_confidence=ClinicalConfidenceOutput.from_dict(sev_conf, device) if sev_conf else None,
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


# ==============================================================================
# CCSM (Clinical Cognitive State Machine) Types
# ==============================================================================


@dataclass
class RouterState:
    """Extensible container for all router-owned recurrent state.

    Uses auxiliary_state dict to support future fields (e.g., adaptive_budget,
    halt_probability) without requiring API or schema changes.
    """
    hidden_state: torch.Tensor            # (Batch, Routing_Hidden_Dim)
    step_index: int                       # Current reasoning depth counter
    cumulative_confidence: torch.Tensor   # (Batch,) running product of step confidences
    routing_history: list[int]            # Block indices selected so far (representative sample)
    auxiliary_state: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate router state fields."""
        if not isinstance(self.hidden_state, torch.Tensor):
            raise RoutingError(f"RouterState hidden_state must be a torch.Tensor, got {type(self.hidden_state)}")
        if len(self.hidden_state.shape) != 2:
            raise RoutingError(f"RouterState hidden_state must be 2D (Batch, H), got {self.hidden_state.shape}")
        if not isinstance(self.cumulative_confidence, torch.Tensor):
            raise RoutingError(f"RouterState cumulative_confidence must be a torch.Tensor, got {type(self.cumulative_confidence)}")
        if self.step_index < 0:
            raise RoutingError(f"RouterState step_index must be non-negative, got {self.step_index}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize router state to dictionary."""
        return {
            "hidden_state": _to_list_if_tensor(self.hidden_state),
            "step_index": self.step_index,
            "cumulative_confidence": _to_list_if_tensor(self.cumulative_confidence),
            "routing_history": list(self.routing_history),
            "auxiliary_state": {k: _to_list_if_tensor(v) for k, v in self.auxiliary_state.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], device: str = "cpu") -> RouterState:
        """Construct RouterState from dictionary."""
        aux = data.get("auxiliary_state", {})
        return cls(
            hidden_state=_to_tensor_if_list(data["hidden_state"], device),
            step_index=data["step_index"],
            cumulative_confidence=_to_tensor_if_list(data["cumulative_confidence"], device),
            routing_history=list(data["routing_history"]),
            auxiliary_state={k: _to_tensor_if_list(v, device) for k, v in aux.items()},
        )


@dataclass(frozen=True)
class ExecutionInstruction:
    """Minimal, opaque instruction consumed by the Execution Engine.

    The Engine never interprets routing semantics. It only sees:
    - which block to execute per sample
    - blending weights for differentiable training
    """
    selected_blocks: torch.Tensor      # (Batch,) int64 — hard block selections
    execution_weights: torch.Tensor    # (Batch, Num_Blocks) — soft blend weights

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate instruction fields."""
        if not isinstance(self.selected_blocks, torch.Tensor):
            raise InterfaceError(f"ExecutionInstruction selected_blocks must be a torch.Tensor, got {type(self.selected_blocks)}")
        if len(self.selected_blocks.shape) != 1:
            raise InterfaceError(f"ExecutionInstruction selected_blocks must be 1D (Batch,), got {self.selected_blocks.shape}")
        if not isinstance(self.execution_weights, torch.Tensor):
            raise InterfaceError(f"ExecutionInstruction execution_weights must be a torch.Tensor, got {type(self.execution_weights)}")
        if len(self.execution_weights.shape) != 2:
            raise InterfaceError(f"ExecutionInstruction execution_weights must be 2D (Batch, Num_Blocks), got {self.execution_weights.shape}")


@dataclass(frozen=True)
class RoutingStepOutput:
    """Single-step routing decision. All tensor fields are batch-aware."""
    routing_logits: torch.Tensor          # (Batch, Num_Blocks)
    routing_probabilities: torch.Tensor   # (Batch, Num_Blocks)
    selected_blocks: torch.Tensor         # (Batch,) int64 per-sample selections
    next_router_state: RouterState
    step_entropy: torch.Tensor            # scalar
    step_confidence: torch.Tensor         # scalar

    def to_execution_instruction(self) -> ExecutionInstruction:
        """Produce the minimal execution instruction from this routing decision."""
        return ExecutionInstruction(
            selected_blocks=self.selected_blocks,
            execution_weights=self.routing_probabilities,
        )

    def validate(self) -> None:
        """Validate step output fields."""
        if not isinstance(self.routing_logits, torch.Tensor) or len(self.routing_logits.shape) != 2:
            raise RoutingError(f"RoutingStepOutput routing_logits must be 2D, got {getattr(self.routing_logits, 'shape', 'N/A')}")
        if not isinstance(self.routing_probabilities, torch.Tensor) or len(self.routing_probabilities.shape) != 2:
            raise RoutingError(f"RoutingStepOutput routing_probabilities must be 2D, got {getattr(self.routing_probabilities, 'shape', 'N/A')}")
        if not isinstance(self.selected_blocks, torch.Tensor) or len(self.selected_blocks.shape) != 1:
            raise RoutingError(f"RoutingStepOutput selected_blocks must be 1D (Batch,), got {getattr(self.selected_blocks, 'shape', 'N/A')}")


# ==============================================================================
# Trace Recording Types
# ==============================================================================


class TraceRecordingLevel(Enum):
    """Preset recording levels for convenience."""
    MINIMAL = "MINIMAL"
    STANDARD = "STANDARD"
    FULL = "FULL"


@dataclass
class TraceRecordingConfig:
    """Fine-grained selective recording flags.

    A recording level sets defaults, then individual flags override.
    """
    level: TraceRecordingLevel = TraceRecordingLevel.STANDARD

    record_hidden_states: bool | None = None
    record_logits: bool | None = None
    record_probabilities: bool | None = None
    record_reasoning_vectors: bool | None = None
    record_entropy: bool | None = None

    def should_record(self, field_name: str) -> bool:
        """Resolve whether a specific field should be recorded.

        Priority: explicit flag > level preset.
        """
        flag_map = {
            "hidden_states": self.record_hidden_states,
            "logits": self.record_logits,
            "probabilities": self.record_probabilities,
            "reasoning_vectors": self.record_reasoning_vectors,
            "entropy": self.record_entropy,
        }
        explicit = flag_map.get(field_name)
        if explicit is not None:
            return explicit

        if self.level == TraceRecordingLevel.MINIMAL:
            return False
        elif self.level == TraceRecordingLevel.STANDARD:
            return field_name in {"entropy", "probabilities"}
        else:  # FULL
            return True


@dataclass
class RoutingStepTrace:
    """Immutable record of a single reasoning step."""
    step_index: int
    selected_block: int                                  # Representative (sample 0)
    selected_blocks_batch: list[int] | None = None       # Full batch selections
    routing_logits: list | None = None                   # Detached, serialized
    routing_probabilities: list | None = None             # Detached, serialized
    routing_entropy: float | None = None
    router_hidden_state: list | None = None              # Detached, serialized
    reasoning_representation: list | None = None          # Detached, serialized
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize step trace to deterministic dictionary."""
        return {
            "step_index": self.step_index,
            "selected_block": self.selected_block,
            "selected_blocks_batch": self.selected_blocks_batch,
            "routing_logits": self.routing_logits,
            "routing_probabilities": self.routing_probabilities,
            "routing_entropy": self.routing_entropy,
            "router_hidden_state": self.router_hidden_state,
            "reasoning_representation": self.reasoning_representation,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RoutingStepTrace:
        """Construct RoutingStepTrace from dictionary."""
        return cls(
            step_index=data["step_index"],
            selected_block=data["selected_block"],
            selected_blocks_batch=data.get("selected_blocks_batch"),
            routing_logits=data.get("routing_logits"),
            routing_probabilities=data.get("routing_probabilities"),
            routing_entropy=data.get("routing_entropy"),
            router_hidden_state=data.get("router_hidden_state"),
            reasoning_representation=data.get("reasoning_representation"),
            confidence=data.get("confidence"),
        )


@dataclass
class RoutingTrace:
    """Complete reasoning trajectory across all steps."""
    steps: list[RoutingStepTrace]
    path_identifier: str
    path_depth: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize routing trace to deterministic dictionary."""
        return {
            "steps": [s.to_dict() for s in self.steps],
            "path_identifier": self.path_identifier,
            "path_depth": self.path_depth,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RoutingTrace:
        """Construct RoutingTrace from dictionary."""
        return cls(
            steps=[RoutingStepTrace.from_dict(s) for s in data["steps"]],
            path_identifier=data["path_identifier"],
            path_depth=data["path_depth"],
        )

    def to_routing_decision(self, device: str = "cpu") -> RoutingDecision:
        """Convert RoutingTrace to a backward-compatible RoutingDecision."""
        depth = self.path_depth
        selected = [s.selected_block for s in self.steps]

        # Reconstruct logits and probabilities tensors from step traces
        step_logits = []
        step_probs = []
        for s in self.steps:
            if s.routing_logits is not None:
                step_logits.append(torch.tensor(s.routing_logits, dtype=torch.float32, device=torch.device(device)))
            if s.routing_probabilities is not None:
                step_probs.append(torch.tensor(s.routing_probabilities, dtype=torch.float32, device=torch.device(device)))

        if step_logits:
            all_logits = torch.stack(step_logits, dim=1)  # (Batch, Depth, Blocks)
        else:
            all_logits = None

        if step_probs:
            all_probs = torch.stack(step_probs, dim=1)
        else:
            all_probs = None

        # Fallback to zero tensors if either is missing, using the other's shape
        if all_logits is None and all_probs is not None:
            all_logits = torch.zeros_like(all_probs)
        elif all_probs is None and all_logits is not None:
            all_probs = torch.zeros_like(all_logits)
        elif all_logits is None and all_probs is None:
            all_logits = torch.zeros(1, depth, 1, device=torch.device(device))
            all_probs = torch.zeros(1, depth, 1, device=torch.device(device))

        avg_entropy = 0.0
        avg_conf = 0.0
        count = 0
        for s in self.steps:
            if s.routing_entropy is not None:
                avg_entropy += s.routing_entropy
                count += 1
            if s.confidence is not None:
                avg_conf += s.confidence
        if count > 0:
            avg_entropy /= count
            avg_conf /= count

        prefix = "ccsm_path_"
        path_id = prefix + "-".join(map(str, selected))

        return RoutingDecision(
            routing_logits=all_logits,
            routing_probabilities=all_probs,
            selected_blocks=selected,
            path_depth=depth,
            routing_entropy=torch.tensor(avg_entropy, device=torch.device(device)),
            routing_confidence=torch.tensor(avg_conf, device=torch.device(device)),
            path_identifier=path_id,
        )


class TraceRecorder:
    """Records reasoning step data according to TraceRecordingConfig."""

    def __init__(self, config: TraceRecordingConfig) -> None:
        self.config = config
        self._steps: list[RoutingStepTrace] = []

    def record(
        self,
        step_output: RoutingStepOutput,
        reasoning_representation: torch.Tensor,
    ) -> None:
        """Record one step's data, respecting recording config."""
        step_idx = step_output.next_router_state.step_index - 1
        selected = int(step_output.selected_blocks[0].item())
        batch_selections = step_output.selected_blocks.detach().cpu().tolist()

        logits_data = None
        if self.config.should_record("logits"):
            logits_data = step_output.routing_logits.detach().cpu().tolist()

        probs_data = None
        if self.config.should_record("probabilities"):
            probs_data = step_output.routing_probabilities.detach().cpu().tolist()

        entropy_data = None
        if self.config.should_record("entropy"):
            entropy_data = float(step_output.step_entropy.detach().item())

        hidden_data = None
        if self.config.should_record("hidden_states"):
            hidden_data = step_output.next_router_state.hidden_state.detach().cpu().tolist()

        reasoning_data = None
        if self.config.should_record("reasoning_vectors"):
            reasoning_data = reasoning_representation.detach().cpu().tolist()

        confidence_data = float(step_output.step_confidence.detach().item())

        trace_step = RoutingStepTrace(
            step_index=step_idx,
            selected_block=selected,
            selected_blocks_batch=batch_selections,
            routing_logits=logits_data,
            routing_probabilities=probs_data,
            routing_entropy=entropy_data,
            router_hidden_state=hidden_data,
            reasoning_representation=reasoning_data,
            confidence=confidence_data,
        )
        self._steps.append(trace_step)

    def finalize(self, path_identifier: str, path_depth: int) -> RoutingTrace:
        """Finalize and return the completed RoutingTrace."""
        return RoutingTrace(
            steps=list(self._steps),
            path_identifier=path_identifier,
            path_depth=path_depth,
        )

    def reset(self) -> None:
        """Clear recorded steps for next forward pass."""
        self._steps = []


@dataclass(frozen=True)
class EvidenceReasoningTrace:
    """Diagnostic telemetry generated by evidence fusion modules."""
    fusion_type: str                   # e.g., "StaticFusion", "AttentionFusion"
    aspect_importance_scores: dict[str, Any] | None = None
    interaction_weights: Any | None = None
    fusion_entropy: float | None = None
    prototype_utilization: dict[str, float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize evidence trace to a deterministic dictionary."""
        def _serialize_tensor_dict(d: dict[str, Any] | None) -> dict[str, Any] | None:
            if not d:
                return None
            return {k: _to_list_if_tensor(v) for k, v in d.items()}

        return {
            "fusion_type": self.fusion_type,
            "aspect_importance_scores": _serialize_tensor_dict(self.aspect_importance_scores),
            "interaction_weights": _to_list_if_tensor(self.interaction_weights),
            "fusion_entropy": self.fusion_entropy,
            "prototype_utilization": self.prototype_utilization,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvidenceReasoningTrace:
        """Construct EvidenceReasoningTrace from dictionary."""
        def _deserialize_tensor_dict(d: dict[str, Any] | None) -> dict[str, Any] | None:
            if not d:
                return None
            return {k: _to_tensor_if_list(v) for k, v in d.items()}

        return cls(
            fusion_type=data.get("fusion_type", "Unknown"),
            aspect_importance_scores=_deserialize_tensor_dict(data.get("aspect_importance_scores")),
            interaction_weights=_to_tensor_if_list(data.get("interaction_weights")),
            fusion_entropy=data.get("fusion_entropy"),
            prototype_utilization=data.get("prototype_utilization"),
            metadata=data.get("metadata", {}),
        )


class EvidenceAttentionRecorder:
    """Records evidence reasoning telemetry during ClinicalEvidenceSynthesizer execution."""
    
    def __init__(self, record_enabled: bool = False) -> None:
        self.record_enabled = record_enabled
        self.current_trace: EvidenceReasoningTrace | None = None

    def record(self, trace: EvidenceReasoningTrace) -> None:
        """Store the generated trace if recording is enabled."""
        if self.record_enabled:
            self.current_trace = trace

    def get_trace(self) -> EvidenceReasoningTrace | None:
        """Retrieve the recorded trace."""
        return self.current_trace

    def reset(self) -> None:
        """Clear recorded trace for next forward pass."""
        self.current_trace = None


# ==============================================================================
# AMCO (Adaptive Multi-Task Clinical Optimization) Types
# ==============================================================================

@dataclass
class OptimizationReasoningTrace:
    """Diagnostic trace for adaptive optimization (AMCO)."""
    optimization_type: str
    effective_task_weights: dict[str, Any] | None = None
    uncertainty_estimates: dict[str, Any] | None = None
    gradient_statistics: dict[str, Any] | None = None
    task_convergence_metrics: dict[str, float] | None = None
    balancing_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize optimization trace to a deterministic dictionary."""
        def _serialize_tensor_dict(d: dict[str, Any] | None) -> dict[str, Any] | None:
            if not d:
                return None
            return {k: _to_list_if_tensor(v) for k, v in d.items()}

        return {
            "optimization_type": self.optimization_type,
            "effective_task_weights": _serialize_tensor_dict(self.effective_task_weights),
            "uncertainty_estimates": _serialize_tensor_dict(self.uncertainty_estimates),
            "gradient_statistics": _serialize_tensor_dict(self.gradient_statistics),
            "task_convergence_metrics": self.task_convergence_metrics,
            "balancing_metadata": self.balancing_metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OptimizationReasoningTrace":
        """Construct OptimizationReasoningTrace from dictionary."""
        def _deserialize_tensor_dict(d: dict[str, Any] | None) -> dict[str, Any] | None:
            if not d:
                return None
            return {k: _to_tensor_if_list(v) for k, v in d.items()}

        return cls(
            optimization_type=data.get("optimization_type", "Unknown"),
            effective_task_weights=_deserialize_tensor_dict(data.get("effective_task_weights")),
            uncertainty_estimates=_deserialize_tensor_dict(data.get("uncertainty_estimates")),
            gradient_statistics=_deserialize_tensor_dict(data.get("gradient_statistics")),
            task_convergence_metrics=data.get("task_convergence_metrics"),
            balancing_metadata=data.get("balancing_metadata", {}),
        )


class OptimizationRecorder:
    """Records optimization telemetry during adaptive loss balancing."""
    
    def __init__(self, record_enabled: bool = False) -> None:
        self.record_enabled = record_enabled
        self.current_trace: OptimizationReasoningTrace | None = None

    def record(self, trace: OptimizationReasoningTrace) -> None:
        """Store the generated trace if recording is enabled."""
        if self.record_enabled:
            self.current_trace = trace

    def get_trace(self) -> OptimizationReasoningTrace | None:
        """Retrieve the recorded trace."""
        return self.current_trace

    def reset(self) -> None:
        """Clear recorded trace for next forward pass."""
        self.current_trace = None


# ==============================================================================
# DCCF (Dynamic Clinical Confidence Framework) Types
# ==============================================================================

@dataclass
class ClinicalConfidenceOutput:
    """Unified clinical confidence output."""
    calibrated_probabilities: torch.Tensor
    confidence_score: torch.Tensor
    uncertainty_score: torch.Tensor
    estimator_metadata: dict[str, Any] = field(default_factory=dict)
    calibration_metadata: dict[str, Any] = field(default_factory=dict)
    future_annotations: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "calibrated_probabilities": _to_list_if_tensor(self.calibrated_probabilities),
            "confidence_score": _to_list_if_tensor(self.confidence_score),
            "uncertainty_score": _to_list_if_tensor(self.uncertainty_score),
            "estimator_metadata": self.estimator_metadata,
            "calibration_metadata": self.calibration_metadata,
            "future_annotations": self.future_annotations,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], device: str = "cpu") -> "ClinicalConfidenceOutput":
        return cls(
            calibrated_probabilities=_to_tensor_if_list(data["calibrated_probabilities"], device),
            confidence_score=_to_tensor_if_list(data["confidence_score"], device),
            uncertainty_score=_to_tensor_if_list(data["uncertainty_score"], device),
            estimator_metadata=data.get("estimator_metadata", {}),
            calibration_metadata=data.get("calibration_metadata", {}),
            future_annotations=data.get("future_annotations", {}),
        )


@dataclass
class ClinicalConfidenceTrace:
    """Diagnostic trace for clinical confidence framework (DCCF)."""
    raw_confidence: torch.Tensor | None = None
    calibrated_confidence: torch.Tensor | None = None
    uncertainty_evolution: dict[str, Any] | None = None
    entropy: torch.Tensor | None = None
    estimator_diagnostics: dict[str, Any] | None = None
    calibration_metadata: dict[str, Any] | None = None
    selective_prediction_metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        def _serialize_tensor_dict(d: dict[str, Any] | None) -> dict[str, Any] | None:
            if not d:
                return None
            return {k: _to_list_if_tensor(v) for k, v in d.items()}

        return {
            "raw_confidence": _to_list_if_tensor(self.raw_confidence),
            "calibrated_confidence": _to_list_if_tensor(self.calibrated_confidence),
            "uncertainty_evolution": _serialize_tensor_dict(self.uncertainty_evolution),
            "entropy": _to_list_if_tensor(self.entropy),
            "estimator_diagnostics": _serialize_tensor_dict(self.estimator_diagnostics),
            "calibration_metadata": _serialize_tensor_dict(self.calibration_metadata),
            "selective_prediction_metadata": _serialize_tensor_dict(self.selective_prediction_metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ClinicalConfidenceTrace":
        def _deserialize_tensor_dict(d: dict[str, Any] | None) -> dict[str, Any] | None:
            if not d:
                return None
            return {k: _to_tensor_if_list(v) for k, v in d.items()}

        return cls(
            raw_confidence=_to_tensor_if_list(data.get("raw_confidence")),
            calibrated_confidence=_to_tensor_if_list(data.get("calibrated_confidence")),
            uncertainty_evolution=_deserialize_tensor_dict(data.get("uncertainty_evolution")),
            entropy=_to_tensor_if_list(data.get("entropy")),
            estimator_diagnostics=_deserialize_tensor_dict(data.get("estimator_diagnostics")),
            calibration_metadata=_deserialize_tensor_dict(data.get("calibration_metadata")),
            selective_prediction_metadata=_deserialize_tensor_dict(data.get("selective_prediction_metadata")),
        )


class ConfidenceRecorder:
    """Records confidence telemetry during DCCF pipeline execution."""
    
    def __init__(self, record_enabled: bool = False) -> None:
        self.record_enabled = record_enabled
        self.current_trace: ClinicalConfidenceTrace | None = None

    def record(self, trace: ClinicalConfidenceTrace) -> None:
        if self.record_enabled:
            self.current_trace = trace

    def get_trace(self) -> ClinicalConfidenceTrace | None:
        return self.current_trace

    def reset(self) -> None:
        self.current_trace = None
