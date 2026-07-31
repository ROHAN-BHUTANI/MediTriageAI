import numpy as np

from src.metrics import classification_report, compute_classification_report


def test_classification_report_normal_multiclass():
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 2, 1, 1, 2]

    report = classification_report(y_true, y_pred, num_classes=3)

    assert "accuracy" in report
    # 5 out of 6 correct
    assert np.isclose(report["accuracy"], 5 / 6)
    assert report["macro_avg"]["precision"] > 0
    assert len(report["per_class"]) == 3


def test_classification_report_partially_supervised_masked():
    # Masked values are -1. sklearn will omit "accuracy" in the raw report.
    y_true = [-1, -1, 0, 1, 2]
    y_pred = [0, 1, 0, 1, 2]  # predictions even for masked rows

    report = classification_report(y_true, y_pred, num_classes=3)

    assert "accuracy" in report
    # Valid indices: [0, 1, 2]. 3 out of 3 correct on VALID indices.
    assert np.isclose(report["accuracy"], 1.0)
    assert len(report["per_class"]) == 3


def test_classification_report_all_masked():
    # All values are masked. Accuracy should be gracefully 0.0.
    y_true = [-1, -1, -1]
    y_pred = [0, 1, 2]

    report = classification_report(y_true, y_pred, num_classes=3)

    assert "accuracy" in report
    assert report["accuracy"] == 0.0
    assert report["macro_avg"]["precision"] == 0.0
    assert report["macro_avg"]["recall"] == 0.0
    assert report["macro_avg"]["f1"] == 0.0


def test_classification_report_single_class():
    y_true = [1, 1, 1]
    y_pred = [1, 1, 1]

    report = classification_report(y_true, y_pred, num_classes=3)

    assert "accuracy" in report
    assert report["accuracy"] == 1.0
    # Class 0 and 2 have support 0
    assert report["per_class"][0]["support"] == 0
    assert report["per_class"][1]["support"] == 3
    assert report["per_class"][2]["support"] == 0


def test_compute_classification_report_missing_accuracy():
    # Same as partially supervised but testing the Iterable[int] API wrapper
    y_true = [-1, 0]
    y_pred = [0, 1]

    report = compute_classification_report(y_true, y_pred, labels=[0, 1])

    assert "accuracy" in report
    # Only 0 is valid, prediction was 1. 0 out of 1 correct.
    assert report["accuracy"] == 0.0
