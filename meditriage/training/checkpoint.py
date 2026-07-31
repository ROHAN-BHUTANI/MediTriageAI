"""Checkpoint Manager for Training Resumption and Model Archiving."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from meditriage.training.config import TrainingConfig
from meditriage.training.utils import get_git_commit_hash


class CheckpointManager:
    """Manages saving, loading, and resuming model training checkpoints."""

    def __init__(self, checkpoint_dir: str | Path):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any | None,
        scaler: torch.cuda.amp.GradScaler | None,
        epoch: int,
        global_step: int,
        config: TrainingConfig,
        metrics: dict[str, Any],
        filename: str = "checkpoint_latest.pt",
    ) -> Path:
        """Save a complete state checkpoint."""
        save_path = self.checkpoint_dir / filename

        # Collect RNG states
        rng_states = {
            "random": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

        checkpoint_data = {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "scaler_state_dict": scaler.state_dict() if scaler else None,
            "rng_states": rng_states,
            "config": config,
            "git_commit": get_git_commit_hash(),
            "metrics": metrics,
        }

        torch.save(checkpoint_data, save_path)
        return save_path

    def load_checkpoint(
        self,
        checkpoint_path: str | Path,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        scaler: torch.cuda.amp.GradScaler | None = None,
    ) -> dict[str, Any]:
        """Load state from checkpoint into model, optimizer, scheduler, and RNGs."""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: '{checkpoint_path}'")

        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Restore model weights
        model.load_state_dict(checkpoint["model_state_dict"])

        # Restore optimizer
        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Restore scheduler
        if scheduler and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"]:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Restore AMP scaler
        if scaler and "scaler_state_dict" in checkpoint and checkpoint["scaler_state_dict"]:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

        # Restore RNG states
        if "rng_states" in checkpoint:
            rngs = checkpoint["rng_states"]
            if "random" in rngs:
                random.setstate(rngs["random"])
            if "numpy" in rngs:
                np.random.set_state(rngs["numpy"])
            if "torch" in rngs:
                torch.set_rng_state(rngs["torch"])

        return {
            "epoch": checkpoint.get("epoch", 0),
            "global_step": checkpoint.get("global_step", 0),
            "config": checkpoint.get("config"),
            "metrics": checkpoint.get("metrics", {}),
            "git_commit": checkpoint.get("git_commit", "unknown"),
        }
