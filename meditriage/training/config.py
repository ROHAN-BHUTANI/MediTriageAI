"""Training Configuration Dataclass and YAML Loader."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TrainingConfig:
    """Production-grade configuration for MediTriageAI model training experiments."""

    # Experiment Identifiers
    experiment_name: str = "meditriage_xlm_roberta_base"
    output_dir: str = "experiments/meditriage_xlm_roberta_base"

    # Model Architecture & Tokenizer Settings
    model_name_or_path: str = "xlm-roberta-base"
    backbone_type: str = "xlm-roberta-base"
    num_triage_classes: int = 5
    num_dept_classes: int = 8
    dropout_rate: float = 0.1
    max_length: int = 128

    # Multi-task Settings
    enable_multitask: bool = True
    triage_loss_weight: float = 1.0
    dept_loss_weight: float = 1.0

    # Optimization Hyperparameters
    batch_size: int = 16
    eval_batch_size: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    warmup_ratio: float = 0.1
    num_epochs: int = 3
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    # Loss & Precision Settings
    loss_type: str = "cross_entropy"  # cross_entropy, focal, weighted_cross_entropy
    focal_gamma: float = 2.0
    use_amp: bool = True  # Automatic Mixed Precision

    # Optimizer & Scheduler Types
    optimizer: str = "adamw"  # adamw, sgd, adam
    scheduler: str = "cosine"  # cosine, linear, onecycle, reducelronplateau

    # Training Loop Controls
    seed: int = 42
    eval_steps: int = 50
    save_steps: int = 100
    logging_steps: int = 10
    early_stopping_patience: int = 3
    save_top_k: int = 1

    # Ablation Experiment Flags
    multilingual_expansion_enabled: bool = True
    linguistic_variation_enabled: bool = True
    phenotype_augmentation_enabled: bool = True
    hard_negatives_enabled: bool = True

    def save(self, path: Path | str) -> None:
        """Save configuration to JSON or YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            if path.suffix in (".yaml", ".yml"):
                yaml.dump(asdict(self), f, default_flow_style=False)
            else:
                json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path | str) -> TrainingConfig:
        """Load configuration from JSON or YAML file."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            if path.suffix in (".yaml", ".yml"):
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        return cls(**data)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrainingConfig:
        cfg = cls()
        for k, v in d.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg
