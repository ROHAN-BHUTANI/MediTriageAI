"""Training Callbacks Framework."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


class Callback:
    """Base callback interface."""

    def on_train_begin(self, logs: dict[str, Any] | None = None) -> None:
        pass

    def on_epoch_begin(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        pass

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        pass

    def on_batch_end(self, step: int, logs: dict[str, Any] | None = None) -> None:
        pass

    def on_train_end(self, logs: dict[str, Any] | None = None) -> None:
        pass


class EarlyStopping(Callback):
    """Early stopping callback on validation metric plateau."""

    def __init__(self, monitor: str = "eval_macro_f1", patience: int = 3, mode: str = "max"):
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.best_score = float("-inf") if mode == "max" else float("inf")
        self.wait = 0
        self.should_stop = False

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        if not logs or self.monitor not in logs:
            return

        current = logs[self.monitor]
        improved = (current > self.best_score) if self.mode == "max" else (current < self.best_score)

        if improved:
            self.best_score = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.should_stop = True


class ModelCheckpoint(Callback):
    """Saves best and latest model checkpoints."""

    def __init__(self, dirpath: str, monitor: str = "eval_macro_f1", save_top_k: int = 1, mode: str = "max"):
        self.dirpath = Path(dirpath)
        self.dirpath.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.save_top_k = save_top_k
        self.mode = mode
        self.best_score = float("-inf") if mode == "max" else float("inf")
        self.best_checkpoint_path: Path | None = None

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        if not logs or self.monitor not in logs:
            return

        current = logs[self.monitor]
        improved = (current > self.best_score) if self.mode == "max" else (current < self.best_score)

        if improved:
            self.best_score = current
            self.best_checkpoint_path = self.dirpath / "best_model.pt"


class LearningRateMonitor(Callback):
    """Logs learning rate history across training steps."""

    def __init__(self):
        self.lr_history: list[float] = []

    def on_batch_end(self, step: int, logs: dict[str, Any] | None = None) -> None:
        if logs and "lr" in logs:
            self.lr_history.append(logs["lr"])


class GradientNormLogger(Callback):
    """Logs total gradient norm across training steps."""

    def __init__(self):
        self.grad_norms: list[float] = []

    def on_batch_end(self, step: int, logs: dict[str, Any] | None = None) -> None:
        if logs and "grad_norm" in logs:
            self.grad_norms.append(logs["grad_norm"])


class PredictionExporter(Callback):
    """Exports test predictions to CSV."""

    def __init__(self, output_file: str | Path):
        self.output_file = Path(output_file)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

    def export(self, df: pd.DataFrame) -> None:
        df.to_csv(self.output_file, index=False)
