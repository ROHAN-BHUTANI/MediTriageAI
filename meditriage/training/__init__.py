"""MediTriageAI Model Training and Experimentation Framework."""

from meditriage.training.callbacks import Callback, EarlyStopping, ModelCheckpoint
from meditriage.training.checkpoint import CheckpointManager
from meditriage.training.config import TrainingConfig
from meditriage.training.experiment import AblationFramework, ExperimentRunner
from meditriage.training.logger import ExperimentLogger
from meditriage.training.losses import FocalLoss, MultiTaskLoss, WeightedCrossEntropyLoss
from meditriage.training.metrics import ClinicalMetricsCalculator
from meditriage.training.optimizer import get_optimizer
from meditriage.training.registry import BACKBONE_REGISTRY, get_backbone_model_id
from meditriage.training.report import generate_experiment_reports
from meditriage.training.scheduler import get_scheduler
from meditriage.training.seed import set_seed
from meditriage.training.trainer import MultiTaskClinicalClassifier, Trainer
from meditriage.training.utils import compute_dataset_fingerprint, get_git_commit_hash, get_hardware_info

__all__ = [
    "TrainingConfig",
    "set_seed",
    "BACKBONE_REGISTRY",
    "get_backbone_model_id",
    "get_hardware_info",
    "get_git_commit_hash",
    "compute_dataset_fingerprint",
    "FocalLoss",
    "WeightedCrossEntropyLoss",
    "MultiTaskLoss",
    "get_optimizer",
    "get_scheduler",
    "ClinicalMetricsCalculator",
    "Callback",
    "EarlyStopping",
    "ModelCheckpoint",
    "ExperimentLogger",
    "CheckpointManager",
    "MultiTaskClinicalClassifier",
    "Trainer",
    "generate_experiment_reports",
    "AblationFramework",
    "ExperimentRunner",
]
