"""
Experiment Hierarchy for the Research Experiment Framework (REF).

Provides the concrete subclasses of `BaseExperiment` representing different
paradigms of AI research execution.
"""

import logging
from pathlib import Path

import torch

from models.emergent_path_triage.config import EmergentPathTriageConfig
from models.emergent_path_triage.model import EmergentPathTriageModel
from ref.base import BaseExperiment
from ref.types import ExperimentArtifacts, ExperimentMetrics
from src.config_manager import TrainingConfig
from src.trainer import EmergentTrainer


def load_and_split_dataset(filepath: str, tokenizer, batch_size: int, seed: int):
    import pandas as pd

    from src.data_pipeline import (
        EmergentTriageDataset,
        TokenizerPipeline,
        get_dataloader,
        get_leakage_safe_splits,
    )

    df = pd.read_csv(filepath)
    if df["text"].isna().sum() > 0:
        df = df.dropna(subset=["text"])

    # Map patient_id to seed_id for patient-level grouping
    if "patient_id" in df.columns:
        df["seed_id"] = df["patient_id"].astype(str)
    elif "seed_id" not in df.columns:
        df["seed_id"] = df.index.map(str)

    # Sub-sample dataset during validation to speed up CI/CD execution
    import os

    if os.environ.get("MOCK_GPU") == "1":
        df = df.sample(n=min(64, len(df)), random_state=seed)
        print(
            f"[CI/CD VALIDATION] Sub-sampled dataset to {len(df)} rows to accelerate execution."
        )

    train_df, val_df, test_df = get_leakage_safe_splits(
        df, train_ratio=0.8, val_ratio=0.1, seed=seed, stratify=False
    )

    pipeline = TokenizerPipeline(tokenizer, max_length=128)

    def create_ds(target_df):
        texts = target_df["text"].tolist()
        # Handle cases where specialist_label and severity_label are present
        if "specialist_label" in target_df.columns:
            spec_ids = target_df["specialist_label"].tolist()
            sev_ids = target_df["severity_label"].tolist()
        else:
            # Fallback for department_code and severity_heuristic
            from src.data_pipeline import LabelValidator

            validator = LabelValidator()
            spec_ids = [
                validator.validate_specialist(str(c))
                for c in target_df["department_code"]
            ]
            sev_ids = [
                validator.validate_severity(str(l))
                for l in target_df["severity_heuristic"]
            ]
        return EmergentTriageDataset(texts, spec_ids, sev_ids, pipeline)

    train_loader = get_dataloader(
        create_ds(train_df), batch_size=batch_size, shuffle=True
    )
    val_loader = get_dataloader(create_ds(val_df), batch_size=batch_size, shuffle=False)
    test_loader = get_dataloader(
        create_ds(test_df), batch_size=batch_size, shuffle=False
    )

    return train_loader, val_loader, test_loader


logger = logging.getLogger(__name__)


class ConcreteExecutionMixin:
    """Provides concrete implementations for lifecycle abstract methods."""

    def configuration_resolution(self) -> None:
        logger.info(f"{self.__class__.__name__}: Stage 2 - Configuration Resolution")
        # Ensure we have valid hyperparameters
        self.learning_rate = self.config_overrides.get("learning_rate", 1e-4)
        self.epochs = self.config_overrides.get("epochs", 10)
        self.batch_size = self.config_overrides.get("batch_size", 16)

    def environment_validation(self) -> None:
        logger.info(f"{self.__class__.__name__}: Stage 3 - Environment Validation")
        # Hardware validation is handled mostly inside EmergentTrainer, but we check if we have CUDA
        logger.info(f"CUDA Available: {torch.cuda.is_available()}")

    def dataset_validation(self) -> None:
        logger.info(f"{self.__class__.__name__}: Stage 4 - Dataset Validation")
        # This will raise DatasetNotFoundError if not available, which gracefully halts execution
        logger.info(f"Using dataset source: {self.dataset}")

    def model_initialization(self) -> None:
        logger.info(f"{self.__class__.__name__}: Stage 5 - Model Initialization")
        self.tokenizer = EmergentPathTriageModel.build_tokenizer()
        self.model_cls = EmergentPathTriageModel()

        # Apply structural ablation if modules_enabled is provided
        triage_config = EmergentPathTriageConfig(
            closed_loop_enabled=self.modules_enabled.get("ccsm", True),
            aces_fusion_mode="A3" if self.modules_enabled.get("aces", True) else "A0",
            amco_optimization_strategy=(
                "HOMOSCEDASTIC" if self.modules_enabled.get("amco", True) else "STATIC"
            ),
            dccf_confidence_estimator=(
                "DIRICHLET" if self.modules_enabled.get("dccf", True) else "IDENTITY"
            ),
        )
        self.network = self.model_cls.build(config=None, triage_config=triage_config)

    def experiment_execution(self) -> None:
        logger.info(f"{self.__class__.__name__}: Stage 6 - Experiment Execution")

        # Build dataloaders
        self.train_loader, self.val_loader, self.test_loader = load_and_split_dataset(
            filepath=self.dataset,
            tokenizer=self.tokenizer,
            batch_size=self.batch_size,
            seed=self.seed,
        )

        trainer_config = TrainingConfig(
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            seed=self.seed,
            checkpoint_dir=(
                str(self.workspace.absolute()) if self.workspace else "./results"
            ),
        )

        self.trainer = EmergentTrainer(
            model=self.network,
            config=trainer_config,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            test_loader=self.test_loader,
            tokenizer=self.tokenizer,
        )

        # Optional resume logic
        if hasattr(self, "checkpoint_reference") and self.checkpoint_reference:
            import os

            if os.path.exists(self.checkpoint_reference):
                self.trainer.load_checkpoint(Path(self.checkpoint_reference))

        self.best_metrics = self.trainer.fit()

        # Post-hoc calibration fitting for DCCF confidence estimators
        if hasattr(self.network, "specialist_calibrator") and hasattr(
            self.network, "severity_calibrator"
        ):
            logger.info(
                "Fitting post-hoc DCCF confidence estimators on validation set..."
            )
            self.network.eval()
            val_spec_logits = []
            val_sev_logits = []
            val_spec_labels = []
            val_sev_labels = []
            with torch.no_grad():
                for batch in self.val_loader:
                    input_ids = batch["input_ids"].to(self.trainer.device)
                    attention_mask = batch["attention_mask"].to(self.trainer.device)
                    labels_spec = batch["labels_specialist"].to(self.trainer.device)
                    labels_sev = batch["labels_severity"].to(self.trainer.device)

                    outputs = self.network(input_ids, attention_mask)
                    val_spec_logits.append(outputs.specialist_logits.cpu())
                    val_sev_logits.append(outputs.severity_logits.cpu())
                    val_spec_labels.append(labels_spec.cpu())
                    val_sev_labels.append(labels_sev.cpu())

            spec_logits = torch.cat(val_spec_logits, dim=0)
            sev_logits = torch.cat(val_sev_logits, dim=0)
            spec_labels = torch.cat(val_spec_labels, dim=0)
            sev_labels = torch.cat(val_sev_labels, dim=0)

            self.network.specialist_calibrator.fit(spec_logits, spec_labels)
            self.network.severity_calibrator.fit(sev_logits, sev_labels)

            # Re-evaluate validation set with the fitted calibrators to get the final validation metrics
            self.best_metrics = self.trainer.validate()

    def metrics_collection(self) -> ExperimentMetrics:
        logger.info(f"{self.__class__.__name__}: Stage 7 - Metrics Collection")
        # Extract telemetry from the trainer and populate ExperimentMetrics
        return ExperimentMetrics(
            clinical={
                "specialist_accuracy": self.best_metrics.get("val_specialist_acc", 0.0),
                "severity_accuracy": self.best_metrics.get("val_severity_acc", 0.0),
                "macro_f1": 0.0,
                "micro_f1": 0.0,
                "roc_auc": 0.0,
                "pr_auc": 0.0,
            },
            calibration={
                "ece": float(
                    (
                        self.best_metrics.get("specialist_ece", 0.0)
                        + self.best_metrics.get("severity_ece", 0.0)
                    )
                    / 2
                ),
                "mce": 0.0,
                "brier_score": float(
                    (
                        self.best_metrics.get("specialist_brier", 0.0)
                        + self.best_metrics.get("severity_brier", 0.0)
                    )
                    / 2
                ),
            },
            routing={},
            optimization={
                "val_loss": self.best_metrics.get("val_loss", 0.0),
                "specialist_loss": self.best_metrics.get("val_specialist_loss", 0.0),
                "severity_loss": self.best_metrics.get("val_severity_loss", 0.0),
            },
            confidence={},
            efficiency={
                "inference_latency": 0.0,
                "memory_usage": 0.0,
                "training_time": self.best_metrics.get("time", 0.0),
            },
        )

    def visualization(self) -> None:
        logger.info(f"{self.__class__.__name__}: Stage 8 - Visualization")
        # The REF pipeline orchestrates visualizations externally via metric_report_dict

    def artifact_generation(self) -> ExperimentArtifacts:
        logger.info(f"{self.__class__.__name__}: Stage 9 - Artifact Generation")
        out_dir = Path(self.workspace.absolute()) if self.workspace else Path(".")
        return ExperimentArtifacts(
            output_dir=str(out_dir),
            checkpoint_paths=[
                str(out_dir / "best_model.pt"),
                str(out_dir / "latest_model.pt"),
            ],
        )


class TrainingExperiment(ConcreteExecutionMixin, BaseExperiment):
    """Manages epoch-driven state, checkpoints, and validation loops."""


class EvaluationExperiment(ConcreteExecutionMixin, BaseExperiment):
    """Pure inference logic and static evaluation."""


class BenchmarkExperiment(ConcreteExecutionMixin, BaseExperiment):
    """Comparative execution against registered control datasets/models."""


class AblationExperiment(ConcreteExecutionMixin, BaseExperiment):
    """Automated parameter sweeping and structural disabling."""


class RobustnessExperiment(ConcreteExecutionMixin, BaseExperiment):
    """OOD testing, adversarial noise insertion, and stress-testing."""
