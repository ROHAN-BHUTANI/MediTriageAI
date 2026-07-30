import pytest
import numpy as np
import torch
import os
from src.metrics import MaskedMetrics

def test_fully_labelled_batches():
    metrics = MaskedMetrics(ignore_index=-1)
    
    spec_logits = torch.tensor([[10.0, 0.0], [0.0, 10.0]])
    sev_logits = torch.tensor([[10.0, 0.0], [0.0, 10.0]])
    
    spec_labels = torch.tensor([0, 1])
    sev_labels = torch.tensor([0, 1])
    
    metrics.update(spec_logits, sev_logits, spec_labels, sev_labels)
    result = metrics.compute(specialist_class_names=["A", "B"], severity_class_names=["S1", "S2"])
    
    assert result["specialist"]["accuracy"] == 1.0
    assert result["severity"]["accuracy"] == 1.0

def test_partially_labelled_batches():
    metrics = MaskedMetrics(ignore_index=-1)
    
    # 3 samples. Middle one is masked.
    spec_logits = torch.tensor([[10.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    sev_logits = torch.tensor([[10.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    
    spec_labels = torch.tensor([0, -1, 1])
    sev_labels = torch.tensor([0, -1, 1])
    
    metrics.update(spec_logits, sev_logits, spec_labels, sev_labels)
    assert len(metrics.specialist_labels) == 2
    assert len(metrics.severity_labels) == 2
    
    result = metrics.compute(specialist_class_names=["A", "B"], severity_class_names=["S1", "S2"])
    assert result["specialist"]["accuracy"] == 1.0

def test_completely_masked_department():
    metrics = MaskedMetrics(ignore_index=-1)
    
    spec_logits = torch.tensor([[10.0, 0.0], [0.0, 10.0]])
    sev_logits = torch.tensor([[10.0, 0.0], [0.0, 10.0]])
    
    spec_labels = torch.tensor([-1, -1])
    sev_labels = torch.tensor([0, 1])
    
    metrics.update(spec_logits, sev_logits, spec_labels, sev_labels)
    assert len(metrics.specialist_labels) == 0
    assert len(metrics.severity_labels) == 2
    
    result = metrics.compute(specialist_class_names=["A", "B"], severity_class_names=["S1", "S2"])
    assert "accuracy" not in result["specialist"]
    assert result["severity"]["accuracy"] == 1.0

def test_completely_masked_severity():
    metrics = MaskedMetrics(ignore_index=-1)
    
    spec_logits = torch.tensor([[10.0, 0.0], [0.0, 10.0]])
    sev_logits = torch.tensor([[10.0, 0.0], [0.0, 10.0]])
    
    spec_labels = torch.tensor([0, 1])
    sev_labels = torch.tensor([-1, -1])
    
    metrics.update(spec_logits, sev_logits, spec_labels, sev_labels)
    assert len(metrics.specialist_labels) == 2
    assert len(metrics.severity_labels) == 0
    
    result = metrics.compute(specialist_class_names=["A", "B"], severity_class_names=["S1", "S2"])
    assert result["specialist"]["accuracy"] == 1.0
    assert "accuracy" not in result["severity"]

def test_mixed_supervision():
    metrics = MaskedMetrics(ignore_index=-1)
    
    spec_logits = torch.tensor([[10.0, 0.0], [10.0, 0.0]])
    sev_logits = torch.tensor([[10.0, 0.0], [10.0, 0.0]])
    
    # One has spec, one has sev
    spec_labels = torch.tensor([0, -1])
    sev_labels = torch.tensor([-1, 0])
    
    metrics.update(spec_logits, sev_logits, spec_labels, sev_labels)
    assert len(metrics.specialist_labels) == 1
    assert len(metrics.severity_labels) == 1

def test_empty_masks():
    metrics = MaskedMetrics(ignore_index=-1)
    
    # 0 size batches
    spec_logits = torch.empty((0, 2))
    sev_logits = torch.empty((0, 2))
    spec_labels = torch.empty((0,), dtype=torch.long)
    sev_labels = torch.empty((0,), dtype=torch.long)
    
    metrics.update(spec_logits, sev_logits, spec_labels, sev_labels)
    result = metrics.compute(specialist_class_names=["A", "B"], severity_class_names=["S1", "S2"])
    assert "accuracy" not in result["specialist"]
    assert "accuracy" not in result["severity"]

def test_auroc_edge_cases():
    metrics = MaskedMetrics(ignore_index=-1)
    
    # Only 1 class present in truth labels
    spec_logits = torch.tensor([[10.0, 0.0], [10.0, 0.0]])
    sev_logits = torch.tensor([[10.0, 0.0], [10.0, 0.0]])
    
    spec_labels = torch.tensor([0, 0])
    sev_labels = torch.tensor([0, 0])
    
    metrics.update(spec_logits, sev_logits, spec_labels, sev_labels)
    result = metrics.compute(specialist_class_names=["A", "B"], severity_class_names=["S1", "S2"])
    
    # AUROC should be NaN since class B is missing
    assert np.isnan(result["specialist"]["auroc"])
    
def test_export(tmp_path):
    metrics = MaskedMetrics(ignore_index=-1)
    spec_logits = torch.tensor([[10.0, 0.0], [0.0, 10.0]])
    sev_logits = torch.tensor([[10.0, 0.0], [0.0, 10.0]])
    spec_labels = torch.tensor([0, 1])
    sev_labels = torch.tensor([0, 1])
    
    metrics.update(spec_logits, sev_logits, spec_labels, sev_labels)
    result = metrics.compute(specialist_class_names=["A", "B"], severity_class_names=["S1", "S2"])
    
    out_dir = tmp_path / "artifacts"
    metrics.export_artifacts(result, str(out_dir), ["A", "B"], ["S1", "S2"])
    
    assert (out_dir / "classification_report.json").exists()
    assert (out_dir / "per_class_metrics.csv").exists()
    assert (out_dir / "specialist_confusion_matrix.csv").exists()
    assert (out_dir / "specialist_normalized_confusion_matrix.csv").exists()
    assert (out_dir / "validation_summary.json").exists()
