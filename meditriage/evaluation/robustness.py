"""Robustness Evaluation Engine across Linguistic and Synthetic Distortions."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


class RobustnessEvaluator:
    """Evaluates clinical model performance under code-switching, Hinglish, typos, and noise."""

    @staticmethod
    def inject_synthetic_typos(text: str, typo_rate: float = 0.1, seed: int = 42) -> str:
        """Inject synthetic character typos into clinical text."""
        rng = random.Random(seed)
        chars = list(text)
        for i in range(len(chars)):
            if chars[i].isalpha() and rng.random() < typo_rate:
                chars[i] = chr(ord(chars[i]) + 1)
        return "".join(chars)

    @classmethod
    def evaluate_linguistic_robustness(
        cls,
        texts: list[str],
        languages: list[str],
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict[str, Any]:
        """Evaluate accuracy and Macro F1 per language/code-switching variant.

        Args:
            texts: List of clinical text strings.
            languages: Corresponding language tags ('en', 'hi', 'hi-Latn', 'hi-en', 'en-hi').
            y_true: True ground-truth labels.
            y_pred: Predicted labels.

        Returns:
            Dictionary of per-language performance metrics.
        """
        results: dict[str, Any] = {}
        unique_langs = set(languages)

        for lang in unique_langs:
            mask = np.array([l == lang for l in languages])
            if np.sum(mask) == 0:
                continue

            sub_true = y_true[mask]
            sub_pred = y_pred[mask]

            acc = float(accuracy_score(sub_true, sub_pred))
            f1 = float(f1_score(sub_true, sub_pred, average="macro", zero_division=0))

            results[lang] = {
                "sample_count": int(np.sum(mask)),
                "accuracy": round(acc, 4),
                "macro_f1": round(f1, 4),
            }

        return results

    @classmethod
    def evaluate_noise_robustness(
        cls,
        y_true: np.ndarray,
        clean_preds: np.ndarray,
        noisy_preds: np.ndarray,
    ) -> dict[str, float]:
        """Compute performance degradation under synthetic noise."""
        clean_f1 = float(f1_score(y_true, clean_preds, average="macro", zero_division=0))
        noisy_f1 = float(f1_score(y_true, noisy_preds, average="macro", zero_division=0))
        drop = clean_f1 - noisy_f1

        return {
            "clean_macro_f1": round(clean_f1, 4),
            "noisy_macro_f1": round(noisy_f1, 4),
            "performance_drop": round(drop, 4),
            "retention_rate": round((noisy_f1 / max(clean_f1, 1e-8)) * 100.0, 2),
        }
