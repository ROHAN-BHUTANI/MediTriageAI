import numpy as np
from meditriage.training.metrics import ClinicalMetricsCalculator
from src.metrics import compute_macro_f1, classification_report

def test_clinical_metrics_calculator_ignore_index():
    # 98.9% -1 labels simulation
    logits = np.array([
        [2.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 2.0, 0.0, 0.0],
        [2.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0, 0.0, 0.0],
    ])
    # 3 masked (-1), 2 valid (0 and 1)
    labels = np.array([-1, -1, -1, 0, 1])

    metrics = ClinicalMetricsCalculator.compute_all_metrics(
        logits, labels, prefix="eval", ignore_index=-1
    )

    # Valid labels are [0, 1], predictions on valid labels are [0, 1] -> perfect accuracy 1.0!
    assert metrics["eval_accuracy"] == 1.0
    assert metrics["eval_macro_f1"] == 1.0

def test_src_metrics_ignore_index():
    y_true = [-1, -1, -1, 0, 1]
    y_pred = [2, 3, 4, 0, 1]

    f1 = compute_macro_f1(y_true, y_pred, "severity")
    assert f1 == 1.0

    report = classification_report(y_true, y_pred, num_classes=5)
    assert report["accuracy"] == 1.0
