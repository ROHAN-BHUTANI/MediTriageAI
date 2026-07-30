"""
Experiment Registry for the Research Experiment Framework (REF).

This module provides a centralized, deterministic registry that tracks
all experiment definitions, configurations, and outputs. It is completely
model-independent.
"""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from ref.types import ExperimentConfiguration, ExperimentMetadata, ExperimentReport


class ExperimentRegistry:
    """Central registry tracking all experiment runs and artifacts."""

    def __init__(self, storage_root: str | Path = "./experiments"):
        self.storage_root = Path(storage_root)
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self._registry_file = self.storage_root / "registry.json"

        # In-memory index: experiment_id -> dict
        self._index: dict[str, dict[str, Any]] = self._load_index()

    def _load_index(self) -> dict[str, dict[str, Any]]:
        """Load the registry index from disk."""
        if self._registry_file.exists():
            with open(self._registry_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_index(self) -> None:
        """Persist the registry index to disk."""
        with open(self._registry_file, "w", encoding="utf-8") as f:
            json.dump(self._index, f, indent=4, sort_keys=True)

    def generate_experiment_id(self, name: str, config_hash: str) -> str:
        """Deterministically generate a unique experiment ID."""
        date_str = datetime.utcnow().strftime("%Y%m%d")
        # Combine name, date, and config hash for uniqueness
        raw = f"{name}_{date_str}_{config_hash}"
        uid = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
        return f"EXP_{date_str}_{name.upper().replace(' ', '_')}_{uid}"

    def register(
        self,
        name: str,
        hypothesis: str,
        dataset: str,
        modules_enabled: dict[str, bool],
        config_overrides: dict[str, Any],
        seed: int,
        checkpoint_reference: str | None = None,
    ) -> tuple[ExperimentMetadata, ExperimentConfiguration, Path]:
        """
        Register a new experiment, producing its metadata, configuration,
        and guaranteeing an isolated output directory.
        """
        config = ExperimentConfiguration(
            config_overrides=config_overrides, checkpoint_reference=checkpoint_reference
        )

        exp_id = self.generate_experiment_id(name, config.configuration_hash)

        if exp_id in self._index:
            logger.warning(
                f"Experiment {exp_id} is already registered. Reusing registration."
            )
            # If the index stored the full report, extract metadata
            metadata_dict = self._index[exp_id].get("metadata", self._index[exp_id])
            return (
                ExperimentMetadata.from_dict(metadata_dict),
                config,
                self.storage_root / exp_id,
            )

        metadata = ExperimentMetadata(
            experiment_id=exp_id,
            experiment_name=name,
            hypothesis=hypothesis,
            modules_enabled=modules_enabled,
            dataset=dataset,
            seed=seed,
        )

        metadata.validate()
        config.validate()

        # Create isolated workspace
        exp_dir = self.storage_root / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Setup standard subdirectories
        for sub in [
            "configs",
            "logs",
            "metrics",
            "plots",
            "tables",
            "predictions",
            "checkpoints",
        ]:
            (exp_dir / sub).mkdir(exist_ok=True)

        # Bind to registry
        self._index[exp_id] = {
            "metadata": metadata.to_dict(),
            "configuration": config.to_dict(),
            "workspace": str(exp_dir.absolute()),
            "status": "REGISTERED",
        }
        self._save_index()

        return metadata, config, exp_dir

    def lookup(self, experiment_id: str) -> dict[str, Any]:
        """Fetch registry metadata for a specific experiment."""
        if experiment_id not in self._index:
            raise KeyError(f"Experiment {experiment_id} not found in registry.")
        return self._index[experiment_id]

    def update_status(
        self, experiment_id: str, status: str, report: ExperimentReport | None = None
    ) -> None:
        """Update the lifecycle status of an experiment and optionally attach the final report."""
        if experiment_id not in self._index:
            raise KeyError(f"Experiment {experiment_id} not found in registry.")

        self._index[experiment_id]["status"] = status

        if report:
            report_dict = report.to_dict()
            self._index[experiment_id]["report"] = report_dict

            # Save report JSON directly into the workspace too
            workspace = Path(self._index[experiment_id]["workspace"])
            with open(workspace / "report.json", "w", encoding="utf-8") as f:
                json.dump(report_dict, f, indent=4, sort_keys=True)

        self._save_index()
