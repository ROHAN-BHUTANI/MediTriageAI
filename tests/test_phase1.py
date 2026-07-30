import pytest
import torch

from src.checkpoint_manager import load_checkpoint, save_checkpoint
from src.config_manager import TrainingConfig


@pytest.fixture
def dummy_config(tmp_path):
    config_path = tmp_path / "test_config.yaml"
    yaml_content = """
learning_rate: 1.0e-4
encoder_lr: 2.0e-5
weight_decay: 0.01
batch_size: 32
epochs: 10
dropout: 0.1
optimizer: "adamw"
scheduler: "cosine"
warmup_ratio: 0.1
loss_weights:
  alpha_specialist: 1.0
  beta_severity: 1.5
gradient_accumulation: 1
gradient_clipping: 1.0
mixed_precision: true
checkpoint_frequency_epochs: 1
early_stopping_patience: 3
early_stopping_metric: "val_loss"
early_stopping_min_improvement: 1.0e-4
seed: 1337
checkpoint_dir: "./test_results"
primary_metric: "department_macro_f1"
encoder_model: "xlm-roberta-base"
"""
    config_path.write_text(yaml_content)
    return config_path


def test_training_config(dummy_config):
    config = TrainingConfig.from_yaml(dummy_config)
    assert config.learning_rate == 1e-4
    assert config.loss_weights["alpha_specialist"] == 1.0
    assert config.get_hash() != ""


def test_checkpoint_integrity(tmp_path):
    ckpt_path = tmp_path / "model.pt"

    # Save checkpoint
    state_dict = {"fc.weight": torch.randn(10, 10)}
    save_checkpoint(
        path=ckpt_path,
        model_short_name="test_model",
        backbone_name="test_backbone",
        config={"test": 1},
        state_dict=state_dict,
        experiment_id="exp_123",
        config_hash="hash_c",
        dataset_manifest_hash="hash_d",
        tokenizer_hash="hash_t",
    )

    # Check if SHA256 was created
    sha_path = tmp_path / "model.pt.sha256"
    assert sha_path.exists()

    # Load checkpoint successfully
    loaded = load_checkpoint(
        ckpt_path,
        expected_config_hash="hash_c",
        expected_dataset_hash="hash_d",
        expected_tokenizer_hash="hash_t",
    )
    assert loaded["version"] == "3.0"
    assert loaded["experiment_id"] == "exp_123"

    # Test mismatch abort
    with pytest.raises(ValueError, match="Resume aborted: Config hash mismatch"):
        load_checkpoint(ckpt_path, expected_config_hash="wrong_hash")

    # Corrupt checksum
    sha_path.write_text("bad_checksum")
    with pytest.raises(RuntimeError, match="Checkpoint corruption detected"):
        load_checkpoint(ckpt_path)
