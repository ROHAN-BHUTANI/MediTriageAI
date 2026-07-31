"""Structured Experiment Logger."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger("meditriage.training")


class ExperimentLogger:
    """Logs training metrics, scalar histories, and experiment artifacts to filesystem."""

    def __init__(self, log_dir: str | Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.history: list[dict[str, Any]] = []

        # Setup file logger
        self.file_handler = logging.FileHandler(self.log_dir / "training.log", encoding="utf-8")
        self.file_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
        logger.addHandler(self.file_handler)
        logger.setLevel(logging.INFO)

    def log_metrics(self, step_or_epoch: int, metrics: dict[str, Any], prefix: str = "train") -> None:
        """Log a dictionary of scalar metrics at a given step/epoch."""
        entry = {"step_or_epoch": step_or_epoch, "prefix": prefix, **metrics}
        self.history.append(entry)

        # Append to history CSV
        df = pd.DataFrame(self.history)
        df.to_csv(self.log_dir / "history.csv", index=False)

        # Write latest metrics JSON
        with open(self.log_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        msg = f"[{prefix.upper()}] Epoch/Step {step_or_epoch} | " + " | ".join(
            f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
            for k, v in metrics.items()
            if not k.endswith("_matrix") and not isinstance(v, (dict, list))
        )
        logger.info(msg)

    def log_experiment_summary(self, summary: dict[str, Any]) -> None:
        """Save master experiment summary JSON."""
        with open(self.log_dir / "experiment_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    def close(self) -> None:
        logger.removeHandler(self.file_handler)
        self.file_handler.close()
