"""Configuration manager for MediTriageAI training pipeline."""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class TrainingConfig:
    learning_rate: float
    encoder_lr: float
    weight_decay: float
    batch_size: int
    epochs: int
    dropout: float
    optimizer: str
    scheduler: str
    warmup_ratio: float
    loss_weights: dict[str, float]
    gradient_accumulation: int
    gradient_clipping: float
    mixed_precision: bool
    checkpoint_frequency_epochs: int
    early_stopping_patience: int
    early_stopping_metric: str
    early_stopping_min_improvement: float
    seed: int
    checkpoint_dir: str
    primary_metric: str
    encoder_model: str

    # Memory efficiency
    dynamic_padding: bool
    gradient_checkpointing: bool
    flash_attention: bool
    pin_memory: bool
    persistent_workers: bool
    prefetch_factor: int
    dataloader_workers: int

    use_torch_compile: bool
    non_blocking_transfers: bool

    _config_dict: dict = field(default_factory=dict, repr=False)
    _config_hash: str = field(default="", repr=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainingConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        # Create a stable hash of the configuration dictionary
        config_str = json.dumps(data, sort_keys=True)
        config_hash = hashlib.sha256(config_str.encode("utf-8")).hexdigest()

        return cls(
            learning_rate=float(data["learning_rate"]),
            encoder_lr=float(data["encoder_lr"]),
            weight_decay=float(data["weight_decay"]),
            batch_size=int(data["batch_size"]),
            epochs=int(data["epochs"]),
            dropout=float(data.get("dropout", 0.1)),
            optimizer=str(data["optimizer"]),
            scheduler=str(data["scheduler"]),
            warmup_ratio=float(data["warmup_ratio"]),
            loss_weights=data["loss_weights"],
            gradient_accumulation=int(data["gradient_accumulation"]),
            gradient_clipping=float(data["gradient_clipping"]),
            mixed_precision=bool(data["mixed_precision"]),
            checkpoint_frequency_epochs=int(data["checkpoint_frequency_epochs"]),
            early_stopping_patience=int(data["early_stopping_patience"]),
            early_stopping_metric=str(data["early_stopping_metric"]),
            early_stopping_min_improvement=float(
                data["early_stopping_min_improvement"]
            ),
            seed=int(data["seed"]),
            checkpoint_dir=str(data["checkpoint_dir"]),
            primary_metric=str(data["primary_metric"]),
            encoder_model=str(data["encoder_model"]),
            dynamic_padding=bool(data.get("dynamic_padding", True)),
            gradient_checkpointing=bool(data.get("gradient_checkpointing", False)),
            flash_attention=bool(data.get("flash_attention", False)),
            pin_memory=bool(data.get("pin_memory", True)),
            persistent_workers=bool(data.get("persistent_workers", True)),
            prefetch_factor=int(data.get("prefetch_factor", 2)),
            dataloader_workers=int(data.get("dataloader_workers", 4)),
            use_torch_compile=bool(data.get("use_torch_compile", False)),
            non_blocking_transfers=bool(data.get("non_blocking_transfers", True)),
            _config_dict=data,
            _config_hash=config_hash,
        )

    def to_dict(self) -> dict:
        return self._config_dict

    def get_hash(self) -> str:
        return self._config_hash
