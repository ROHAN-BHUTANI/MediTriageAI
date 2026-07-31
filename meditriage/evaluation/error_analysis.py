"""Clinical Error Analysis Engine."""

from __future__ import annotations

from typing import Any

import numpy as np


class ClinicalErrorAnalyzer:
    """Analyzes false positives, false negatives, specialty confusion, and calibration errors."""

    @classmethod
    def analyze_errors(
        cls,
        texts: list[str],
        y_true: np.ndarray,
        y_pred: np.ndarray,
        probs: np.ndarray,
        class_names: list[str] | None = None,
        high_conf_threshold: float = 0.80,
    ) -> dict[str, Any]:
        """Perform comprehensive clinical error analysis.

        Args:
            texts: Original input clinical texts.
            y_true: Ground truth class indices.
            y_pred: Predicted class indices.
            probs: Prediction probabilities matrix (N, C).
            class_names: Class label names.
            high_conf_threshold: Probability threshold for high-confidence errors.

        Returns:
            Dictionary containing categorized error analysis metrics.
        """
        n = len(y_true)
        if n == 0:
            return {}

        confidences = np.max(probs, axis=1) if probs.size > 0 else np.ones(n)
        errors_mask = (y_pred != y_true)

        total_errors = int(np.sum(errors_mask))
        error_rate = round(float(total_errors / max(n, 1)), 4)

        # High-confidence errors
        high_conf_errors = []
        low_conf_errors = []

        for idx in range(n):
            if errors_mask[idx]:
                true_name = class_names[y_true[idx]] if class_names and y_true[idx] < len(class_names) else str(y_true[idx])
                pred_name = class_names[y_pred[idx]] if class_names and y_pred[idx] < len(class_names) else str(y_pred[idx])
                conf = float(confidences[idx])

                item = {
                    "text": texts[idx] if idx < len(texts) else "",
                    "true_label": true_name,
                    "predicted_label": pred_name,
                    "confidence": round(conf, 4),
                }

                if conf >= high_conf_threshold:
                    high_conf_errors.append(item)
                else:
                    low_conf_errors.append(item)

        # Most confused class pairs
        pair_counts: dict[str, int] = {}
        for idx in range(n):
            if errors_mask[idx]:
                true_c = class_names[y_true[idx]] if class_names and y_true[idx] < len(class_names) else str(y_true[idx])
                pred_c = class_names[y_pred[idx]] if class_names and y_pred[idx] < len(class_names) else str(y_pred[idx])
                pair_key = f"{true_c} -> {pred_c}"
                pair_counts[pair_key] = pair_counts.get(pair_key, 0) + 1

        sorted_confused = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)

        return {
            "total_samples": n,
            "total_errors": total_errors,
            "error_rate": error_rate,
            "high_confidence_errors_count": len(high_conf_errors),
            "high_confidence_errors": high_conf_errors[:10],
            "top_confused_class_pairs": dict(sorted_confused[:5]),
        }
