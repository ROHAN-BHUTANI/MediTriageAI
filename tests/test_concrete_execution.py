import pytest
import pandas as pd
import torch
import os
from pathlib import Path
from unittest.mock import patch

from src.data_ingestion import (
    load_and_split_dataset, 
    SchemaValidationError, 
    DatasetNotFoundError,
    TriageDataset
)
from ref.experiments import TrainingExperiment
from ref.registry import ExperimentRegistry
from models.emergent_path_triage.model import EmergentPathTriageModel

@pytest.fixture
def mock_dataset_df():
    return pd.DataFrame({
        'patient_id': [1, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'text': ['Patient has fever'] * 12,
        'specialist_label': [0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        'severity_label': [0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    })

def test_graceful_dataset_not_found():
    """Verify DatasetNotFoundError is raised when file is missing."""
    tokenizer = EmergentPathTriageModel.build_tokenizer()
    with pytest.raises(DatasetNotFoundError):
        load_and_split_dataset("non_existent_file.csv", tokenizer)

def test_schema_validation(tmp_path):
    """Verify SchemaValidationError on invalid dataframe."""
    df = pd.DataFrame({'wrong_col': [1, 2]})
    file_path = tmp_path / "dummy.csv"
    df.to_csv(file_path, index=False)
    
    tokenizer = EmergentPathTriageModel.build_tokenizer()
    with pytest.raises(SchemaValidationError):
        load_and_split_dataset(file_path, tokenizer)

def test_dataloader_validation(mock_dataset_df, tmp_path):
    """Verify Dataloader works and returns correct tensors."""
    file_path = tmp_path / "dummy.csv"
    mock_dataset_df.to_csv(file_path, index=False)
    
    tokenizer = EmergentPathTriageModel.build_tokenizer()
    
    train_dl, val_dl, test_dl = load_and_split_dataset(file_path, tokenizer, batch_size=2)
    
    batch = next(iter(train_dl))
    assert 'input_ids' in batch
    assert 'attention_mask' in batch
    assert 'labels_specialist' in batch
    assert 'labels_severity' in batch
    assert batch['input_ids'].dim() == 2

def test_concrete_experiment_lifecycle(mock_dataset_df, tmp_path):
    """Verify full forward, backward, optimizer step, checkpoint, and telemetry emission."""
    file_path = tmp_path / "dummy.csv"
    mock_dataset_df.to_csv(file_path, index=False)
    
    # We use tmp_path for the registry and outputs
    registry = ExperimentRegistry(storage_root=tmp_path)
    
    experiment = TrainingExperiment(
        name="concrete_smoke_test",
        hypothesis="Testing concrete execution",
        dataset=str(file_path),
        modules_enabled={"ccsm": True, "aces": False, "amco": False, "dccf": False},
        config_overrides={"epochs": 1, "batch_size": 2, "use_amp": False},
        seed=42,
        registry=registry
    )
    
    # Override output directory logic for isolated tests
    experiment.workspace = tmp_path / "outputs"
    experiment.workspace.mkdir(parents=True, exist_ok=True)
    
    # Execute the lifecycle manually instead of running entire campaign runner
    experiment.metadata, _, _ = experiment.experiment_registration()
    experiment.configuration_resolution()
    experiment.environment_validation()
    experiment.dataset_validation()
    experiment.model_initialization()
    experiment.experiment_execution()
    metrics = experiment.metrics_collection()
    artifacts = experiment.artifact_generation()
    
    # Validation Checks
    # 1. Telemetry emission
    assert metrics.optimization["val_loss"] is not None
    assert metrics.clinical["specialist_accuracy"] is not None
    
    # 2. Checkpoints generated
    assert (experiment.workspace / "best_model.pt").exists()
    assert (experiment.workspace / "latest_model.pt").exists()
    
    # 3. Artifact paths in return
    assert str(experiment.workspace / "best_model.pt") in artifacts.checkpoint_paths
