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
    BaseStepRouter,
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
from models.emergent_path_triage.dces import (
    ClinicalEvidenceSynthesizer,
    BaseEvidenceFusion,
    StaticFusion,
    AttentionFusion
)
from models.emergent_path_triage.dcrr import ClinicalReasoningRouter
from models.emergent_path_triage.ctb import ClinicalThoughtBlock
from models.emergent_path_triage.engine import ReasoningPathExecutionEngine, ClinicalThoughtExecutionEngine
from models.emergent_path_triage.compat import LegacyExecutionEngineAdapter
from models.emergent_path_triage.heads import PredictionHead
from models.emergent_path_triage.dcp import DynamicConsistencyProjection
from models.emergent_path_triage.aces_utils import EvidenceDiagnostics
from models.emergent_path_triage.amco import BaseLossBalancer, StaticLossBalancer, HomoscedasticBalancer, GradNormBalancer
from models.emergent_path_triage.amco_utils import OptimizationDiagnostics
from models.emergent_path_triage.dccf import BaseConfidenceEstimator, IdentityEstimator, TemperatureScalingEstimator, VectorScalingEstimator, DirichletEstimator
from models.emergent_path_triage.dccf_utils import ClinicalConfidenceDiagnostics
from models.emergent_path_triage.types import (
    EvidenceRepresentation,
    RoutingDecision,
    RouterState,
    ExecutionInstruction,
    RoutingStepOutput,
    RoutingStepTrace,
    RoutingTrace,
    TraceRecordingLevel,
    TraceRecordingConfig,
    TraceRecorder,
    ThoughtPath,
    ModelOutputs,
    AuxiliaryLosses,
    EvidenceReasoningTrace,
    EvidenceAttentionRecorder,
    OptimizationReasoningTrace,
    OptimizationRecorder,
    ClinicalConfidenceOutput,
    ClinicalConfidenceTrace,
    ConfidenceRecorder,
)
from models.emergent_path_triage.amco import (
    BaseLossBalancer,
    StaticLossBalancer,
    HomoscedasticBalancer,
    GradNormBalancer,
)
from models.emergent_path_triage.amco_utils import OptimizationDiagnostics



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
    "BaseStepRouter",
    "BaseClinicalThoughtBlock",
    "BaseConsistencyProjection",
    "BaseEmergentPathTriage",
    "BaseCheckpointRegistry",
    "EvidenceRepresentation",
    "RoutingDecision",
    "RouterState",
    "ExecutionInstruction",
    "RoutingStepOutput",
    "RoutingStepTrace",
    "RoutingTrace",
    "TraceRecordingLevel",
    "TraceRecordingConfig",
    "TraceRecorder",
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
    "ClinicalThoughtExecutionEngine",
    "LegacyExecutionEngineAdapter",
    "PredictionHead",
    "DynamicConsistencyProjection",
    "BaseEvidenceFusion",
    "StaticFusion",
    "AttentionFusion",
    "EvidenceReasoningTrace",
    "EvidenceAttentionRecorder",
    "EvidenceDiagnostics",
    "OptimizationReasoningTrace",
    "OptimizationRecorder",
    "BaseLossBalancer",
    "StaticLossBalancer",
    "HomoscedasticBalancer",
    "GradNormBalancer",
    "OptimizationDiagnostics",
    "BaseConfidenceEstimator",
    "IdentityEstimator",
    "TemperatureScalingEstimator",
    "VectorScalingEstimator",
    "DirichletEstimator",
    "ClinicalConfidenceOutput",
    "ClinicalConfidenceTrace",
    "ConfidenceRecorder",
    "ClinicalConfidenceDiagnostics",
]
