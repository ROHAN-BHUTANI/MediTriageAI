"""Emergent Path-Aligned Co-evolutionary Reasoning Network (E-PATH-CO-REASON)."""

from __future__ import annotations

from models.emergent_path_triage.aces_utils import EvidenceDiagnostics
from models.emergent_path_triage.amco import (
    BaseLossBalancer,
    HomoscedasticBalancer,
    StaticLossBalancer,
)
from models.emergent_path_triage.amco_utils import OptimizationDiagnostics
from models.emergent_path_triage.compat import LegacyExecutionEngineAdapter
from models.emergent_path_triage.config import EmergentPathTriageConfig
from models.emergent_path_triage.constants import (
    DEFAULT_LATENT_DIM,
    DEFAULT_NUM_CTBS,
    DEFAULT_ROUTING_DEPTH,
    DEFAULT_TEMPERATURE,
    NUM_SEVERITY_LABELS,
    NUM_SPECIALIST_CLASSES,
)
from models.emergent_path_triage.ctb import ClinicalThoughtBlock
from models.emergent_path_triage.dccf import (
    BaseConfidenceEstimator,
    DirichletEstimator,
    IdentityEstimator,
    TemperatureScalingEstimator,
    VectorScalingEstimator,
)
from models.emergent_path_triage.dccf_utils import ClinicalConfidenceDiagnostics
from models.emergent_path_triage.dces import (
    AttentionFusion,
    BaseEvidenceFusion,
    ClinicalEvidenceSynthesizer,
    StaticFusion,
)
from models.emergent_path_triage.dcp import DynamicConsistencyProjection
from models.emergent_path_triage.dcrr import ClinicalReasoningRouter
from models.emergent_path_triage.engine import (
    ClinicalThoughtExecutionEngine,
    ReasoningPathExecutionEngine,
)
from models.emergent_path_triage.exceptions import (
    CompatibilityError,
    ConfigurationError,
    InterfaceError,
    MediTriageError,
    RoutingError,
)
from models.emergent_path_triage.heads import PredictionHead
from models.emergent_path_triage.hooks import apply_loss_hook
from models.emergent_path_triage.interfaces import (
    BaseCheckpointRegistry,
    BaseClinicalEvidenceSynthesizer,
    BaseClinicalThoughtBlock,
    BaseConsistencyProjection,
    BaseEmergentPathTriage,
    BaseReasoningRouter,
    BaseStepRouter,
)
from models.emergent_path_triage.model import (
    EmergentPathCheckpointRegistry,
    EmergentPathTriageModel,
    EmergentPathTriageTransformer,
)
from models.emergent_path_triage.types import (
    AuxiliaryLosses,
    ClinicalConfidenceOutput,
    ClinicalConfidenceTrace,
    ConfidenceRecorder,
    EvidenceAttentionRecorder,
    EvidenceReasoningTrace,
    EvidenceRepresentation,
    ExecutionInstruction,
    ModelOutputs,
    OptimizationReasoningTrace,
    OptimizationRecorder,
    RouterState,
    RoutingDecision,
    RoutingStepOutput,
    RoutingStepTrace,
    RoutingTrace,
    ThoughtPath,
    TraceRecorder,
    TraceRecordingConfig,
    TraceRecordingLevel,
)

__all__ = [
    "DEFAULT_LATENT_DIM",
    "DEFAULT_NUM_CTBS",
    "DEFAULT_ROUTING_DEPTH",
    "DEFAULT_TEMPERATURE",
    "NUM_SEVERITY_LABELS",
    "NUM_SPECIALIST_CLASSES",
    "AttentionFusion",
    "AuxiliaryLosses",
    "BaseCheckpointRegistry",
    "BaseClinicalEvidenceSynthesizer",
    "BaseClinicalThoughtBlock",
    "BaseConfidenceEstimator",
    "BaseConsistencyProjection",
    "BaseEmergentPathTriage",
    "BaseEvidenceFusion",
    "BaseLossBalancer",
    "BaseReasoningRouter",
    "BaseStepRouter",
    "ClinicalConfidenceDiagnostics",
    "ClinicalConfidenceOutput",
    "ClinicalConfidenceTrace",
    "ClinicalEvidenceSynthesizer",
    "ClinicalReasoningRouter",
    "ClinicalThoughtBlock",
    "ClinicalThoughtExecutionEngine",
    "CompatibilityError",
    "ConfidenceRecorder",
    "ConfigurationError",
    "DirichletEstimator",
    "DynamicConsistencyProjection",
    "EmergentPathCheckpointRegistry",
    "EmergentPathTriageConfig",
    "EmergentPathTriageModel",
    "EmergentPathTriageTransformer",
    "EvidenceAttentionRecorder",
    "EvidenceDiagnostics",
    "EvidenceReasoningTrace",
    "EvidenceRepresentation",
    "ExecutionInstruction",
    "HomoscedasticBalancer",
    "IdentityEstimator",
    "InterfaceError",
    "LegacyExecutionEngineAdapter",
    "MediTriageError",
    "ModelOutputs",
    "OptimizationDiagnostics",
    "OptimizationReasoningTrace",
    "OptimizationRecorder",
    "PredictionHead",
    "ReasoningPathExecutionEngine",
    "RouterState",
    "RoutingDecision",
    "RoutingError",
    "RoutingStepOutput",
    "RoutingStepTrace",
    "RoutingTrace",
    "StaticFusion",
    "StaticLossBalancer",
    "TemperatureScalingEstimator",
    "ThoughtPath",
    "TraceRecorder",
    "TraceRecordingConfig",
    "TraceRecordingLevel",
    "VectorScalingEstimator",
    "apply_loss_hook",
]
