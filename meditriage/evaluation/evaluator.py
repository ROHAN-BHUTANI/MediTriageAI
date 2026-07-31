"""Research Evaluator Master Module."""

from __future__ import annotations

from typing import Any

import numpy as np

from meditriage.evaluation.error_analysis import ClinicalErrorAnalyzer
from meditriage.evaluation.robustness import RobustnessEvaluator
from meditriage.evaluation.significance import StatisticalSignificanceEngine
from meditriage.training.metrics import ClinicalMetricsCalculator


class ResearchEvaluator:
    """Master evaluator combining clinical metrics, robustness, error analysis, and significance."""

    @classmethod
    def evaluate_model(
        cls,
        texts: list[str],
        y_true: np.ndarray,
        y_pred: np.ndarray,
        probs: np.ndarray,
        languages: list[str] | None = None,
        class_names: list[str] | None = None,
        y_pred_baseline: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Perform comprehensive scientific evaluation of a clinical classification model."""
        # 1. Base Clinical Metrics
        base_metrics = ClinicalMetricsCalculator.compute_all_metrics(
            logits=np.log(np.clip(probs, 1e-8, 1.0)),
            labels=y_true,
            class_names=class_names,
            prefix="research",
        )

        # 2. Error Analysis
        err_analysis = ClinicalErrorAnalyzer.analyze_errors(
            texts=texts,
            y_true=y_true,
            y_pred=y_pred,
            probs=probs,
            class_names=class_names,
        )

        # 3. Bootstrap 95% Confidence Interval
        mean_f1, ci_low, ci_high = (
            StatisticalSignificanceEngine.bootstrap_confidence_interval(
                y_true, y_pred, metric="macro_f1", num_bootstraps=500
            )
        )
        sig_data = {"mean": mean_f1, "ci_lower": ci_low, "ci_upper": ci_high}

        # 4. Paired Significance Test against Baseline (if provided)
        if y_pred_baseline is not None:
            paired_res = StatisticalSignificanceEngine.paired_significance_test(
                y_true, y_pred_baseline, y_pred
            )
            sig_data.update(paired_res)

        # 5. Robustness evaluation per language
        robustness_data = {}
        if languages:
            robustness_data = RobustnessEvaluator.evaluate_linguistic_robustness(
                texts, languages, y_true, y_pred
            )

        return {
            "metrics": base_metrics,
            "error_analysis": err_analysis,
            "significance": sig_data,
            "robustness": robustness_data,
        }
