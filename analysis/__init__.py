"""MediTriageAI Error Analysis Library.

This package exposes the public modular APIs of the error analysis framework, 
allowing researchers to reuse individual modules for future medical NLP experiments.
"""

from __future__ import annotations

from analysis.config import config, AnalysisConfig
from analysis.io import generate_and_cache_predictions, load_test_dataframe
from analysis.language_detector import HeuristicLanguageDetector, BaseLanguageDetector
from analysis.metrics import (
    bootstrap_metric_ci,
    compute_prediction_entropy,
    compute_top_k_accuracy,
    compute_overall_metrics,
    compute_per_class_metrics,
    add_confidence_columns
)
from analysis.calibration import (
    compute_ece_mce,
    compute_nll,
    compute_brier_score,
    get_reliability_curve_data
)
from analysis.agreement import compute_pairwise_agreement
from analysis.taxonomy import classify_errors, generate_taxonomy_summary
from analysis.utils import set_seed, compute_mcnemar_test, compute_sha256, df_to_markdown

__all__ = [
    "config",
    "AnalysisConfig",
    "generate_and_cache_predictions",
    "load_test_dataframe",
    "HeuristicLanguageDetector",
    "BaseLanguageDetector",
    "bootstrap_metric_ci",
    "compute_prediction_entropy",
    "compute_top_k_accuracy",
    "compute_overall_metrics",
    "compute_per_class_metrics",
    "add_confidence_columns",
    "compute_ece_mce",
    "compute_nll",
    "compute_brier_score",
    "get_reliability_curve_data",
    "compute_pairwise_agreement",
    "classify_errors",
    "generate_taxonomy_summary",
    "set_seed",
    "compute_mcnemar_test",
    "compute_sha256",
    "df_to_markdown",
]

