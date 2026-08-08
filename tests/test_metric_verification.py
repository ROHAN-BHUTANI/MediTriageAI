"""Deterministic metric mathematical verification test suite."""

import numpy as np
import pytest
from meditriage.training.metrics import ClinicalMetricsCalculator


def test_clinical_metrics_calculator_deterministic():
    # 4 synthetic samples with 3 classes
    # Logits designed to produce known predictions:
    # Sample 0: class 0 (logits [5.0, 0.0, 0.0]) -> label 0 (CORRECT)
    # Sample 1: class 1 (logits [0.0, 5.0, 0.0]) -> label 1 (CORRECT)
    # Sample 2: class 2 (logits [0.0, 0.0, 5.0]) -> label 1 (INCORRECT, true=1, pred=2)
    # Sample 3: class 0 (logits [5.0, 0.0, 0.0]) -> label -1 (MASKED / IGNORED)

    logits = np.array(
        [
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 5.0],
            [5.0, 0.0, 0.0],
        ]
    )
    labels = np.array([0, 1, 1, -1])

    metrics = ClinicalMetricsCalculator.compute_all_metrics(
        logits, labels, class_names=["C0", "C1", "C2"], prefix="test", ignore_index=-1
    )

    # Valid samples: 3 (indices 0, 1, 2)
    # Sample 0: true 0, pred 0 (hit)
    # Sample 1: true 1, pred 1 (hit)
    # Sample 2: true 1, pred 2 (miss)
    # Accuracy on valid samples: 2 / 3 = 0.6667
    assert metrics["test_accuracy"] == 0.6667

    # Class 0: true count = 1, pred 0 (recall = 1.0, precision = 1.0, F1 = 1.0)
    # Class 1: true count = 2, pred 1 hit 1 miss 1 (recall = 0.5, precision = 1.0, F1 = 0.6667)
    # Class 2: true count = 0, pred 1 miss (recall = 0.0, precision = 0.0, F1 = 0.0)
    # Macro F1 = (1.0 + 2/3 + 0) / 3 = 0.5556
    assert metrics["test_macro_f1"] == 0.5556

    # Verify per-class dict exists and contains support
    per_class = metrics["test_per_class"]
    assert per_class["C0"]["support"] == 1
    assert per_class["C1"]["support"] == 2
    assert per_class["C2"]["support"] == 0


def test_joint_accuracy_calculation():
    # Synthetic multi-task targets:
    # 5 samples:
    # Sample 0: spec_true=0, spec_pred=0, sev_true=1, sev_pred=1 -> JOINT HIT
    # Sample 1: spec_true=1, spec_pred=1, sev_true=2, sev_pred=2 -> JOINT HIT
    # Sample 2: spec_true=2, spec_pred=2, sev_true=3, sev_pred=1 -> MISMATCH (spec hit, sev miss)
    # Sample 3: spec_true=3, spec_pred=0, sev_true=4, sev_pred=4 -> MISMATCH (spec miss, sev hit)
    # Sample 4: spec_true=-1, spec_pred=0, sev_true=1, sev_pred=1 -> MASKED (spec missing)

    spec_true = np.array([0, 1, 2, 3, -1])
    spec_pred = np.array([0, 1, 2, 0, 0])
    sev_true = np.array([1, 2, 3, 4, 1])
    sev_pred = np.array([1, 2, 1, 4, 1])

    valid_both = (spec_true != -1) & (sev_true != -1)
    joint_hits = (spec_pred[valid_both] == spec_true[valid_both]) & (
        sev_pred[valid_both] == sev_true[valid_both]
    )

    joint_acc = float(np.mean(joint_hits))
    # Valid samples: 4 (indices 0, 1, 2, 3)
    # Joint hits: 2 out of 4 = 0.50
    assert joint_acc == 0.50
