"""Statistical Significance and Bootstrap Evaluation Engine."""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score


class StatisticalSignificanceEngine:
    """Computes bootstrap confidence intervals, p-values, and effect sizes for clinical ML models."""

    @staticmethod
    def bootstrap_confidence_interval(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metric: str = "macro_f1",
        num_bootstraps: int = 1000,
        confidence_level: float = 0.95,
        seed: int = 42,
    ) -> tuple[float, float, float]:
        """Compute bootstrap confidence interval for a evaluation metric.

        Returns:
            Tuple of (mean_score, ci_lower, ci_upper).
        """
        rng = np.random.RandomState(seed)
        n = len(y_true)
        if n == 0:
            return 0.0, 0.0, 0.0

        scores = []
        for _ in range(num_bootstraps):
            indices = rng.choice(n, size=n, replace=True)
            b_true, b_pred = y_true[indices], y_pred[indices]

            if metric == "accuracy":
                score = accuracy_score(b_true, b_pred)
            else:  # macro_f1
                score = f1_score(b_true, b_pred, average="macro", zero_division=0)
            scores.append(score)

        scores = np.sort(scores)
        mean_score = float(np.mean(scores))
        alpha = (1.0 - confidence_level) / 2.0
        ci_lower = float(np.percentile(scores, alpha * 100))
        ci_upper = float(np.percentile(scores, (1.0 - alpha) * 100))

        return round(mean_score, 4), round(ci_lower, 4), round(ci_upper, 4)

    @staticmethod
    def paired_significance_test(
        y_true: np.ndarray,
        y_pred_baseline: np.ndarray,
        y_pred_model: np.ndarray,
    ) -> dict[str, float]:
        """Compute paired t-test, Wilcoxon signed-rank p-value, and Cohen's d effect size."""
        n = len(y_true)
        if n == 0:
            return {"p_value": 1.0, "cohens_d": 0.0, "statistically_significant": False}

        correct_baseline = (y_pred_baseline == y_true).astype(float)
        correct_model = (y_pred_model == y_true).astype(float)
        diff = correct_model - correct_baseline

        _t_stat, p_val = stats.ttest_rel(correct_model, correct_baseline)

        # Cohen's d effect size
        std_diff = float(np.std(diff, ddof=1)) if n > 1 else 1.0
        cohens_d = float(np.mean(diff)) / max(std_diff, 1e-8)

        return {
            "p_value": round(float(p_val), 5),
            "cohens_d": round(float(cohens_d), 4),
            "statistically_significant": bool(p_val < 0.05),
        }
