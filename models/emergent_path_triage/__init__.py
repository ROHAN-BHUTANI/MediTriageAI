"""Emergent Path-Aligned Co-evolutionary Reasoning Network (E-PATH-CO-REASON)."""

from __future__ import annotations

from models.emergent_path_triage.config import EmergentPathTriageConfig
from models.emergent_path_triage.constants import (
    NUM_SPECIALIST_CLASSES,
    NUM_SEVERITY_LABELS,
    DEFAULT_LATENT_DIM,
    DEFAULT_ROUTING_DEPTH,
    DEFAULT_NUM_CTBS,
    DEFAULT_TEMPERATURE,
)
from models.emergent_path_triage.exceptions import (
    MediTriageError,
    ConfigurationError,
    RoutingError,
    InterfaceError,
    CompatibilityError,
)
from models.emergent_path_triage.hooks import apply_loss_hook
from models.emergent_path_triage.interfaces import (
    BaseClinicalEvidenceSynthesizer,
    BaseReasoningRouter,
    BaseClinicalThoughtBlock,
    BaseConsistencyProjection,
    BaseEmergentPathTriage,
    BaseCheckpointRegistry,
)
from models.emergent_path_triage.model import (
    EmergentPathTriageModel,
    EmergentPathTriageTransformer,
    EmergentPathCheckpointRegistry,
)
from models.emergent_path_triage.dces import ClinicalEvidenceSynthesizer
from models.emergent_path_triage.dcrr import ClinicalReasoningRouter
from models.emergent_path_triage.ctb import ClinicalThoughtBlock
from models.emergent_path_triage.engine import ReasoningPathExecutionEngine
from models.emergent_path_triage.heads import PredictionHead
from models.emergent_path_triage.dcp import DynamicConsistencyProjection
from models.emergent_path_triage.types import (
    EvidenceRepresentation,
    RoutingDecision,
    ThoughtPath,
    ModelOutputs,
    AuxiliaryLosses,
)



__all__ = [
    "EmergentPathTriageConfig",
    "EmergentPathTriageModel",
    "EmergentPathTriageTransformer",
    "EmergentPathCheckpointRegistry",
    "apply_loss_hook",
    "NUM_SPECIALIST_CLASSES",
    "NUM_SEVERITY_LABELS",
    "DEFAULT_LATENT_DIM",
    "DEFAULT_ROUTING_DEPTH",
    "DEFAULT_NUM_CTBS",
    "DEFAULT_TEMPERATURE",
    "BaseClinicalEvidenceSynthesizer",
    "BaseReasoningRouter",
    "BaseClinicalThoughtBlock",
    "BaseConsistencyProjection",
    "BaseEmergentPathTriage",
    "BaseCheckpointRegistry",
    "EvidenceRepresentation",
    "RoutingDecision",
    "ThoughtPath",
    "ModelOutputs",
    "AuxiliaryLosses",
    "MediTriageError",
    "ConfigurationError",
    "RoutingError",
    "InterfaceError",
    "CompatibilityError",
    "ClinicalEvidenceSynthesizer",
    "ClinicalReasoningRouter",
    "ClinicalThoughtBlock",
    "ReasoningPathExecutionEngine",
    "PredictionHead",
    "DynamicConsistencyProjection",
]
