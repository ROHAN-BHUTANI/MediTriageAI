"""Optimizer Factory for Model Training."""

from __future__ import annotations

import torch
from torch import nn
from torch.optim import SGD, Adam, AdamW

from meditriage.training.config import TrainingConfig


def get_optimizer(model: nn.Module, cfg: TrainingConfig) -> torch.optim.Optimizer:
    """Build optimizer with weight decay exclusion for bias and normalization parameters.

    Args:
        model: PyTorch model module.
        cfg: Training configuration.

    Returns:
        Configured PyTorch Optimizer instance.
    """
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": cfg.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": 0.0,
        },
    ]

    opt_name = cfg.optimizer.lower()
    if opt_name == "adamw":
        return AdamW(optimizer_grouped_parameters, lr=cfg.learning_rate)
    elif opt_name == "adam":
        return Adam(optimizer_grouped_parameters, lr=cfg.learning_rate)
    elif opt_name == "sgd":
        return SGD(optimizer_grouped_parameters, lr=cfg.learning_rate, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: '{opt_name}'")
