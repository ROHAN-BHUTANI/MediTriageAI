"""Regression tests for AMP GradScaler overflow recovery and numerical stability."""
from unittest.mock import MagicMock
import pytest
import torch
from torch import nn
from torch.amp import GradScaler

from src.config_manager import TrainingConfig
from src.trainer import EmergentTrainer
import models.emergent_path_triage.hooks


class SimpleLinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(8, 8)
        self.head = nn.Linear(8, 2)

    def forward(self, input_ids, attention_mask):
        # input_ids: (batch_size, 8)
        x = self.encoder(input_ids.float())
        spec_logits = torch.randn(len(input_ids), 13, device=input_ids.device)
        sev_logits = torch.randn(len(input_ids), 5, device=input_ids.device)
        return spec_logits, sev_logits


@pytest.fixture
def base_config():
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
        mixed_precision=True,
        checkpoint_frequency_epochs=1,
        early_stopping_patience=1,
        early_stopping_metric="val_loss",
        early_stopping_min_improvement=1e-4,
        seed=42,
        checkpoint_dir="./test_amp_results",
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


def test_amp_overflow_recovery_skips_step_without_raising(base_config):
    """Test that simulated FP16 gradient overflow reduces scale and skips optimizer step without fatal exception."""
    model = SimpleLinearModel()
    dummy_batches = [
        {
            "input_ids": torch.randn(2, 8),
            "attention_mask": torch.ones(2, 8),
            "labels_specialist": torch.tensor([0, 1]),
            "labels_severity": torch.tensor([0, 1]),
        },
        {
            "input_ids": torch.randn(2, 8),
            "attention_mask": torch.ones(2, 8),
            "labels_specialist": torch.tensor([0, 1]),
            "labels_severity": torch.tensor([0, 1]),
        },
    ]

    trainer = EmergentTrainer(
        model=model,
        config=base_config,
        train_loader=dummy_batches,
        val_loader=MagicMock(),
    )
    trainer.use_amp = True
    trainer.scaler = GradScaler(trainer.device.type, enabled=True, init_scale=65536.0)

    # Inject an Inf gradient in batch 0
    orig_hook = models.emergent_path_triage.hooks.apply_loss_hook
    step_count = 0

    def mock_hook(model, spec_logits, sev_logits, labels_spec, labels_sev, loss_fn):
        nonlocal step_count
        step_count += 1
        loss = loss_fn(spec_logits, sev_logits, labels_spec, labels_sev)
        # Multiply by weight parameter so loss has gradient connection
        return {"joint_loss": loss["joint_loss"] + 0.0 * model.encoder.weight.sum()}

    models.emergent_path_triage.hooks.apply_loss_hook = mock_hook

    # Manually make gradient Inf in the first step backward
    initial_scale = trainer.scaler.get_scale()
    assert initial_scale == 65536.0

    # Hook backward to inject Inf on step 0
    def inject_inf_backward_hook(grad):
        if step_count == 1:
            g = grad.clone()
            g[0, 0] = float("inf")
            return g
        return grad

    hook_handle = model.encoder.weight.register_hook(inject_inf_backward_hook)

    try:
        metrics = trainer.train_epoch(epoch=1)
        # Should complete without raising RuntimeError
        assert "loss" in metrics
        # Scale should have been reduced due to Inf in batch 0
        final_scale = trainer.scaler.get_scale()
        assert final_scale < initial_scale
    finally:
        hook_handle.remove()
        models.emergent_path_triage.hooks.apply_loss_hook = orig_hook


def test_fp32_gradient_instability_raises_runtime_error(base_config):
    """Test that non-AMP (FP32) training still raises RuntimeError if gradients contain NaN/Inf."""
    base_config.mixed_precision = False
    model = SimpleLinearModel()
    dummy_batches = [
        {
            "input_ids": torch.randn(2, 8),
            "attention_mask": torch.ones(2, 8),
            "labels_specialist": torch.tensor([0, 1]),
            "labels_severity": torch.tensor([0, 1]),
        }
    ]

    trainer = EmergentTrainer(
        model=model,
        config=base_config,
        train_loader=dummy_batches,
        val_loader=MagicMock(),
    )
    trainer.use_amp = False

    orig_hook = models.emergent_path_triage.hooks.apply_loss_hook

    def mock_hook(model, spec_logits, sev_logits, labels_spec, labels_sev, loss_fn):
        loss = loss_fn(spec_logits, sev_logits, labels_spec, labels_sev)
        return {"joint_loss": loss["joint_loss"] + 0.0 * model.encoder.weight.sum()}

    models.emergent_path_triage.hooks.apply_loss_hook = mock_hook

    def inject_inf_backward_hook(grad):
        g = grad.clone()
        g[0, 0] = float("inf")
        return g

    hook_handle = model.encoder.weight.register_hook(inject_inf_backward_hook)

    try:
        with pytest.raises(RuntimeError, match="Numerical Stability Error! NaN or Inf in gradients"):
            trainer.train_epoch(epoch=1)
    finally:
        hook_handle.remove()
        models.emergent_path_triage.hooks.apply_loss_hook = orig_hook


def test_loss_nan_inf_aborts_in_all_modes(base_config):
    """Test that NaN/Inf in loss aborts immediately across all modes."""
    model = SimpleLinearModel()
    dummy_batches = [
        {
            "input_ids": torch.randn(2, 8),
            "attention_mask": torch.ones(2, 8),
            "labels_specialist": torch.tensor([0, 1]),
            "labels_severity": torch.tensor([0, 1]),
        }
    ]

    trainer = EmergentTrainer(
        model=model,
        config=base_config,
        train_loader=dummy_batches,
        val_loader=MagicMock(),
    )

    orig_hook = models.emergent_path_triage.hooks.apply_loss_hook

    def mock_nan_loss(*args, **kwargs):
        return {"joint_loss": torch.tensor(float("nan"))}

    models.emergent_path_triage.hooks.apply_loss_hook = mock_nan_loss

    try:
        with pytest.raises(RuntimeError, match="Numerical Stability Error! Loss is NaN or Inf"):
            trainer.train_epoch(epoch=1)
    finally:
        models.emergent_path_triage.hooks.apply_loss_hook = orig_hook
