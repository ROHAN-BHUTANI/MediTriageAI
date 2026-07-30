from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from src.config_manager import TrainingConfig
from src.trainer import EmergentTrainer


class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(10, 10)
        self.head = nn.Linear(10, 2)

    def forward(self, input_ids, attention_mask):
        return torch.randn(2, 13), torch.randn(2, 5)


@pytest.fixture
def dummy_config():
    return TrainingConfig(
        learning_rate=1e-4,
        encoder_lr=2e-5,
        weight_decay=0.01,
        batch_size=2,
        epochs=1,
        dropout=0.1,
        optimizer="adamw",
        scheduler="cosine",
        warmup_ratio=0.1,
        loss_weights={"alpha_specialist": 1.0, "beta_severity": 1.0},
        gradient_accumulation=1,
        gradient_clipping=1.0,
        mixed_precision=False,
        checkpoint_frequency_epochs=1,
        early_stopping_patience=1,
        early_stopping_metric="val_loss",
        early_stopping_min_improvement=1e-4,
        seed=1337,
        checkpoint_dir="./test_results",
        primary_metric="val_loss",
        encoder_model="mock",
        dynamic_padding=True,
        gradient_checkpointing=False,
        flash_attention=False,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=2,
        dataloader_workers=0,
        use_torch_compile=False,
        non_blocking_transfers=True,
    )


def test_trainer_initialization(dummy_config):
    model = MockModel()
    trainer = EmergentTrainer(
        model=model,
        config=dummy_config,
        train_loader=MagicMock(),
        val_loader=MagicMock(),
    )
    assert trainer.optimizer is not None
    assert trainer.scheduler is not None


def test_numerical_stability_abort(dummy_config):
    model = MockModel()
    trainer = EmergentTrainer(
        model=model,
        config=dummy_config,
        train_loader=[
            {
                "input_ids": torch.randint(0, 100, (2, 10)),
                "attention_mask": torch.ones(2, 10),
                "labels_specialist": torch.tensor([0, 1]),
                "labels_severity": torch.tensor([0, 1]),
            }
        ],
        val_loader=MagicMock(),
    )

    # Mock loss hook to return NaN
    import models.emergent_path_triage.hooks

    original_hook = models.emergent_path_triage.hooks.apply_loss_hook

    def mock_loss_hook(*args, **kwargs):
        return {"joint_loss": torch.tensor(float("nan"))}

    models.emergent_path_triage.hooks.apply_loss_hook = mock_loss_hook

    with pytest.raises(
        RuntimeError, match="Numerical Stability Error! Loss is NaN or Inf"
    ):
        trainer.train_epoch(epoch=1)

    # Restore
    models.emergent_path_triage.hooks.apply_loss_hook = original_hook
