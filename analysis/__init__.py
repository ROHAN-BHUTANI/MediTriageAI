"""MediTriageAI Error Analysis Library.

This package exposes the public modular APIs of the error analysis framework,
allowing researchers to reuse individual modules for future medical NLP experiments.
"""

from __future__ import annotations

from analysis.agreement import compute_pairwise_agreement
from analysis.calibration import (
    compute_brier_score,
    compute_ece_mce,
    compute_nll,
    get_reliability_curve_data,
)
from analysis.config import AnalysisConfig, config
from analysis.io import generate_and_cache_predictions, load_test_dataframe
from analysis.language_detector import BaseLanguageDetector, HeuristicLanguageDetector
from analysis.metrics import (
    add_confidence_columns,
    bootstrap_metric_ci,
    compute_overall_metrics,
    compute_per_class_metrics,
    compute_prediction_entropy,
    compute_top_k_accuracy,
)
from analysis.taxonomy import classify_errors, generate_taxonomy_summary
from analysis.utils import (
    compute_mcnemar_test,
    compute_sha256,
    df_to_markdown,
    set_seed,
)

__all__ = [
    "AnalysisConfig",
    "BaseLanguageDetector",
    "HeuristicLanguageDetector",
    "add_confidence_columns",
    "bootstrap_metric_ci",
    "classify_errors",
    "compute_brier_score",
    "compute_ece_mce",
    "compute_mcnemar_test",
    "compute_nll",
    "compute_overall_metrics",
    "compute_pairwise_agreement",
    "compute_per_class_metrics",
    "compute_prediction_entropy",
    "compute_sha256",
    "compute_top_k_accuracy",
    "config",
    "df_to_markdown",
    "generate_and_cache_predictions",
    "generate_taxonomy_summary",
    "get_reliability_curve_data",
    "load_test_dataframe",
    "set_seed",
]
