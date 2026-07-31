"""Learning Rate Scheduler Factory."""

from __future__ import annotations

from typing import Any

import torch
from torch.optim.lr_scheduler import (
    OneCycleLR,
    ReduceLROnPlateau,
)
from transformers import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from meditriage.training.config import TrainingConfig


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: TrainingConfig,
    num_training_steps: int,
) -> Any:
    """Build learning rate scheduler.

    Args:
        optimizer: PyTorch Optimizer instance.
        cfg: Training configuration.
        num_training_steps: Total training steps across all epochs.

    Returns:
        Configured PyTorch or Transformers LRScheduler instance.
    """
    sched_name = cfg.scheduler.lower()

    if sched_name == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=cfg.warmup_steps,
            num_training_steps=max(num_training_steps, 1),
        )
    elif sched_name == "linear":
        return get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=cfg.warmup_steps,
            num_training_steps=max(num_training_steps, 1),
        )
    elif sched_name == "onecycle":
        return OneCycleLR(
            optimizer,
            max_lr=cfg.learning_rate,
            total_steps=max(num_training_steps, 1),
        )
    elif sched_name == "reducelronplateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=2,
        )
    else:
        return get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=cfg.warmup_steps,
            num_training_steps=max(num_training_steps, 1),
        )
