from __future__ import annotations

import numpy as np

from src.metrics import (
    cohens_kappa,
    compute_classification_report,
    compute_macro_f1,
    compute_per_class_f1,
    compute_ordinal_confusion,
    generate_latex_table,
    generate_novelty_summary,
    landis_koch_label,
    severity_metrics,
)


def test_compute_macro_f1_perfect_and_partial() -> None:
    assert compute_macro_f1([0, 1, 2], [0, 1, 2], "specialist") == 1.0
    score = compute_macro_f1([0, 1, 2], [0, 2, 1], "severity")
    assert 0.0 <= score <= 1.0


def test_compute_ordinal_confusion_counts() -> None:
    result = compute_ordinal_confusion([0, 1, 2, 3], [0, 2, 1, 4])
    assert result["exact_match"] == 1
    assert result["adjacent_confusion"] == 3
    assert result["distant_confusion"] == 0


def test_compute_classification_report_structure() -> None:
    report = compute_classification_report([0, 0, 1, 1], [0, 1, 1, 1], labels=[0, 1])
    assert set(report) == {"accuracy", "macro_avg", "weighted_avg", "per_class"}
    assert len(report["per_class"]) == 2
    assert report["per_class"][0]["class"] == "0"


def test_kappa_landis_koch_benchmark() -> None:
    assert landis_koch_label(-0.1) == "poor"
    assert landis_koch_label(0.10) == "slight"
    assert landis_koch_label(0.30) == "fair"
    assert landis_koch_label(0.50) == "moderate"
    assert landis_koch_label(0.70) == "substantial"
    assert landis_koch_label(0.90) == "almost perfect"
    assert cohens_kappa([0, 0, 1, 1], [0, 0, 1, 1], 2) == 1.0


def test_severity_metrics_contains_ordinal_fields() -> None:
    report = severity_metrics([0, 1, 2, 3, 4], [0, 2, 1, 3, 4])
    assert report["per_class"][0]["class"] == "S1"
    assert "ordinal_confusion" in report
    assert "ordinal_error_matrix" in report
    assert report["mean_absolute_error"] >= 0.0


def test_generate_novelty_summary_uses_actual_numbers() -> None:
    summary = generate_novelty_summary(
        {
            "xlm": {"model_display_name": "XLM-RoBERTa-large", "is_novel_contribution": True, "specialist_macro_f1": 0.90, "severity_macro_f1": 0.80},
            "mbert": {"model_display_name": "mBERT", "specialist_macro_f1": 0.70, "severity_macro_f1": 0.60},
        }
    )
    assert "0.200" in summary


def test_compute_per_class_f1_and_latex_table() -> None:
    per_class = compute_per_class_f1([0, 1, 1, 0], [0, 1, 0, 0], ["alpha", "beta"])
    assert set(per_class) == {"alpha", "beta"}
    table = generate_latex_table(
        {
            "xlm": {"model_display_name": "XLM-RoBERTa-large", "specialist_macro_f1": 0.9, "severity_macro_f1": 0.8},
            "mbert": {"model_display_name": "mBERT", "specialist_macro_f1": 0.7, "severity_macro_f1": 0.6},
        },
        "specialist",
    )
    assert "\\begin{tabular}" in table


def test_compute_macro_f1_accepts_numpy_arrays() -> None:
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    assert 0.0 <= compute_macro_f1(y_true, y_pred, "specialist") <= 1.0
