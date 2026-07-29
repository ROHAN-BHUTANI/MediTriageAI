"""
Base Experiment lifecycle for the Research Experiment Framework (REF).

Provides the abstract `BaseExperiment` class which strictly dictates
the 10-stage execution lifecycle of any research experiment.
"""

from abc import ABC, abstractmethod
from typing import Any
from pathlib import Path
import logging

from ref.types import (
    ExperimentMetadata,
    ExperimentConfiguration,
    ExperimentMetrics,
    ExperimentArtifacts,
    ExperimentSummary,
    ExperimentReport
)
from ref.registry import ExperimentRegistry

logger = logging.getLogger(__name__)

class BaseExperiment(ABC):
    """
    Model-agnostic abstract base class for all research experiments.
    Enforces a strict 10-stage lifecycle. No stage may combine multiple responsibilities.
    """

    def __init__(self, registry: ExperimentRegistry, name: str, hypothesis: str, dataset: str, modules_enabled: dict[str, bool], config_overrides: dict[str, Any], seed: int, checkpoint_reference: str | None = None):
        self.registry = registry
        self.name = name
        self.hypothesis = hypothesis
        self.dataset = dataset
        self.modules_enabled = modules_enabled
        self.config_overrides = config_overrides
        self.seed = seed
        self.checkpoint_reference = checkpoint_reference
        
        # State populated during the lifecycle
        self.metadata: ExperimentMetadata | None = None
        self.configuration: ExperimentConfiguration | None = None
        self.workspace: Path | None = None
        self.metrics: ExperimentMetrics | None = None
        self.artifacts: ExperimentArtifacts | None = None
        self.summary: ExperimentSummary | None = None
        self.report: ExperimentReport | None = None

    def execute_lifecycle(self) -> ExperimentReport:
        """
        The immutable 10-stage template method.
        Executes the strict progression of an experiment.
        """
        try:
            logger.info(f"Starting experiment lifecycle for: {self.name}")
            
            # Stage 1
            self.metadata, self.configuration, self.workspace = self.experiment_registration()
            self.registry.update_status(self.metadata.experiment_id, "CONFIGURING")
            
            # Stage 2
            self.configuration_resolution()
            self.registry.update_status(self.metadata.experiment_id, "VALIDATING_ENV")
            
            # Stage 3
            self.environment_validation()
            self.registry.update_status(self.metadata.experiment_id, "VALIDATING_DATASET")
            
            # Stage 4
            self.dataset_validation()
            self.registry.update_status(self.metadata.experiment_id, "INITIALIZING_MODEL")
            
            # Stage 5
            self.model_initialization()
            self.registry.update_status(self.metadata.experiment_id, "EXECUTING")
            
            # Stage 6
            self.experiment_execution()
            self.registry.update_status(self.metadata.experiment_id, "COLLECTING_METRICS")
            
            # Stage 7
            self.metrics = self.metrics_collection()
            self.registry.update_status(self.metadata.experiment_id, "VISUALIZING")
            
            # Stage 8
            self.visualization()
            self.registry.update_status(self.metadata.experiment_id, "GENERATING_ARTIFACTS")
            
            # Stage 9
            self.artifacts = self.artifact_generation()
            self.registry.update_status(self.metadata.experiment_id, "FINALIZING_REPORT")
            
            # Stage 10
            self.report = self.experiment_report()
            
            self.registry.update_status(self.metadata.experiment_id, "COMPLETED", report=self.report)
            logger.info(f"Experiment {self.name} completed successfully.")
            return self.report
            
        except Exception as e:
            logger.error(f"Experiment {self.name} failed during lifecycle: {e}", exc_info=True)
            if self.metadata:
                self.registry.update_status(self.metadata.experiment_id, "FAILED")
            raise

    # =========================================================================
    # STAGE 1: Experiment Registration
    # =========================================================================
    def experiment_registration(self) -> tuple[ExperimentMetadata, ExperimentConfiguration, Path]:
        """Registers the experiment in the registry and constructs isolated outputs."""
        logger.info("Stage 1: Experiment Registration")
        return self.registry.register(
            name=self.name,
            hypothesis=self.hypothesis,
            dataset=self.dataset,
            modules_enabled=self.modules_enabled,
            config_overrides=self.config_overrides,
            seed=self.seed,
            checkpoint_reference=self.checkpoint_reference
        )

    # =========================================================================
    # STAGE 2: Configuration Resolution
    # =========================================================================
    @abstractmethod
    def configuration_resolution(self) -> None:
        """Parses logic to bind overrides and lock hyperparameters."""

    # =========================================================================
    # STAGE 3: Environment Validation
    # =========================================================================
    @abstractmethod
    def environment_validation(self) -> None:
        """Profiles hardware and validates environment boundaries."""

    # =========================================================================
    # STAGE 4: Dataset Validation
    # =========================================================================
    @abstractmethod
    def dataset_validation(self) -> None:
        """Hashes and maps dataset partitions."""

    # =========================================================================
    # STAGE 5: Model Initialization
    # =========================================================================
    @abstractmethod
    def model_initialization(self) -> None:
        """Instantiates architectures and loads pre-trained weights."""

    # =========================================================================
    # STAGE 6: Experiment Execution
    # =========================================================================
    @abstractmethod
    def experiment_execution(self) -> None:
        """Runs the core isolated training or inference loop."""

    # =========================================================================
    # STAGE 7: Metrics Collection
    # =========================================================================
    @abstractmethod
    def metrics_collection(self) -> ExperimentMetrics:
        """Aggregates execution tensors into strict numeric metric types."""

    # =========================================================================
    # STAGE 8: Visualization
    # =========================================================================
    @abstractmethod
    def visualization(self) -> None:
        """Renders graphical insights derived from metrics offline."""

    # =========================================================================
    # STAGE 9: Artifact Generation
    # =========================================================================
    @abstractmethod
    def artifact_generation(self) -> ExperimentArtifacts:
        """Saves physical artifacts (checkpoints, tables) and returns pointers."""

    # =========================================================================
    # STAGE 10: Experiment Report
    # =========================================================================
    def experiment_report(self) -> ExperimentReport:
        """Finalizes the summary and structured complete report."""
        logger.info("Stage 10: Experiment Report")
        if not all([self.metadata, self.configuration, self.metrics, self.artifacts]):
            raise ValueError("Cannot generate report: missing upstream components.")
            
        self.summary = ExperimentSummary(
            status="COMPLETED",
            conclusion=f"Experiment '{self.name}' completed the lifecycle.",
            key_metrics={},
            warnings=[]
        )
        
        report = ExperimentReport(
            metadata=self.metadata,
            configuration=self.configuration,
            metrics=self.metrics,
            artifacts=self.artifacts,
            summary=self.summary
        )
        report.validate()
        return report
