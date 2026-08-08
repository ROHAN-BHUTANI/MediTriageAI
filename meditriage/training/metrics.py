"""Comprehensive Clinical Classification Metrics Calculator."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


class ClinicalMetricsCalculator:
    """Calculates comprehensive evaluation metrics for medical classification."""

    @staticmethod
    def compute_calibration_error(
        probs: np.ndarray, labels: np.ndarray, num_bins: int = 10
    ) -> float:
        """Compute Expected Calibration Error (ECE)."""
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        accuracies = (predictions == labels).astype(float)

        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        ece = 0.0

        for i in range(num_bins):
            bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = np.mean(in_bin)

            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(accuracies[in_bin])
                avg_confidence_in_bin = np.mean(confidences[in_bin])
                ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin

        return round(float(ece), 4)

    @classmethod
    def compute_all_metrics(
        cls,
        logits: np.ndarray,
        labels: np.ndarray,
        class_names: list[str] | None = None,
        prefix: str = "eval",
        ignore_index: int = -1,
    ) -> dict[str, Any]:
        """Compute full suite of clinical classification metrics.

        Args:
            logits: Logits array of shape (N, C).
            labels: Ground truth labels array of shape (N,).
            class_names: Optional class name labels.
            prefix: Metric key prefix.
            ignore_index: Target label index to ignore (e.g. -1 for unmapped labels).

        Returns:
            Dictionary of calculated metrics.
        """
        if len(logits) == 0 or len(labels) == 0:
            return {}

        # Mask out ignored labels (e.g. -1 unmapped targets)
        if ignore_index is not None:
            valid_mask = labels != ignore_index
            logits = logits[valid_mask]
            labels = labels[valid_mask]

        num_classes = (
            logits.shape[1]
            if len(logits) > 0 and len(logits.shape) > 1
            else (len(class_names) if class_names else 0)
        )

        if len(logits) == 0 or len(labels) == 0:
            per_class_empty = {}
            if class_names:
                for c_name in class_names:
                    per_class_empty[c_name] = {
                        "precision": 0.0,
                        "recall": 0.0,
                        "f1": 0.0,
                        "support": 0,
                    }
            return {
                f"{prefix}_accuracy": 0.0,
                f"{prefix}_balanced_accuracy": 0.0,
                f"{prefix}_macro_precision": 0.0,
                f"{prefix}_macro_recall": 0.0,
                f"{prefix}_macro_f1": 0.0,
                f"{prefix}_weighted_f1": 0.0,
                f"{prefix}_top2_accuracy": 0.0,
                f"{prefix}_calibration_error": 0.0,
                f"{prefix}_confusion_matrix": [],
                f"{prefix}_per_class": per_class_empty,
            }

        # Convert logits to probabilities via softmax
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        preds = np.argmax(probs, axis=1)

        labels_list = list(range(num_classes)) if num_classes > 0 else None

        acc = float(accuracy_score(labels, preds))
        balanced_acc = float(balanced_accuracy_score(labels, preds))

        # Precision, Recall, F1
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
            labels, preds, labels=labels_list, average="macro", zero_division=0
        )
        _p_weighted, _r_weighted, f1_weighted, _ = precision_recall_fscore_support(
            labels, preds, labels=labels_list, average="weighted", zero_division=0
        )

        # Top-K Accuracy (Top-2)
        top2_acc = 0.0
        if probs.shape[1] >= 2:
            top2_preds = np.argsort(probs, axis=1)[:, -2:]
            top2_hits = [labels[i] in top2_preds[i] for i in range(len(labels))]
            top2_acc = float(np.mean(top2_hits))

        # Calibration Error
        ece = cls.compute_calibration_error(probs, labels)

        # Confusion Matrix
        cm = confusion_matrix(labels, preds, labels=labels_list).tolist()

        # Per-class metrics
        p_per_class, r_per_class, f1_per_class, support_per_class = (
            precision_recall_fscore_support(
                labels, preds, labels=labels_list, average=None, zero_division=0
            )
        )

        per_class_metrics = {}
        for idx in range(len(p_per_class)):
            c_name = (
                class_names[idx]
                if class_names and idx < len(class_names)
                else f"class_{idx}"
            )
            per_class_metrics[c_name] = {
                "precision": round(float(p_per_class[idx]), 4),
                "recall": round(float(r_per_class[idx]), 4),
                "f1": round(float(f1_per_class[idx]), 4),
                "support": int(support_per_class[idx]),
            }

        return {
            f"{prefix}_accuracy": round(acc, 4),
            f"{prefix}_balanced_accuracy": round(balanced_acc, 4),
            f"{prefix}_macro_precision": round(float(p_macro), 4),
            f"{prefix}_macro_recall": round(float(r_macro), 4),
            f"{prefix}_macro_f1": round(float(f1_macro), 4),
            f"{prefix}_weighted_f1": round(float(f1_weighted), 4),
            f"{prefix}_top2_accuracy": round(top2_acc, 4),
            f"{prefix}_calibration_error": ece,
            f"{prefix}_confusion_matrix": cm,
            f"{prefix}_per_class": per_class_metrics,
        }

