"""Error taxonomy classification and failure analysis for the MediTriageAI analysis framework."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.model import SPECIALIST_CLASSES


def classify_errors(df: pd.DataFrame, config: Any) -> pd.DataFrame:
    """Analyze predictions and flag taxonomy categories for each sample.

    Args:
        df: Predictions DataFrame, already enriched with confidence metrics.
        config: AnalysisConfig instance.

    Returns:
        DataFrame containing taxonomy flags.
    """
    df = df.copy()

    # Base correctness flags
    df["spec_correct"] = df["true_specialist"] == df["pred_specialist"]
    df["sev_correct"] = df["true_severity"] == df["pred_severity"]

    # 1. Wrong specialist but correct severity
    df["tax_wrong_spec_correct_sev"] = ~df["spec_correct"] & df["sev_correct"]

    # 2. Correct specialist but wrong severity
    df["tax_correct_spec_wrong_sev"] = df["spec_correct"] & ~df["sev_correct"]

    # 3. Both incorrect
    df["tax_both_incorrect"] = ~df["spec_correct"] & ~df["sev_correct"]

    # 4. Near-miss specialist (true specialist is in top-3 predicted classes, but prediction is wrong)
    near_misses = []
    for _, row in df.iterrows():
        if row["spec_correct"]:
            near_misses.append(False)
            continue
        probs = np.array(row["specialist_probabilities"])
        true_idx = SPECIALIST_CLASSES.index(row["true_specialist"])
        top3_indices = np.argsort(probs)[-3:]
        near_misses.append(true_idx in top3_indices)
    df["tax_near_miss_specialist"] = near_misses

    # 5. High-confidence wrong prediction (any incorrect head prediction with confidence > threshold)
    high_conf_wrong = []
    for _, row in df.iterrows():
        spec_wrong_high = (not row["spec_correct"]) and (
            row["specialist_top1_conf"] > config.high_confidence_threshold
        )
        sev_wrong_high = (not row["sev_correct"]) and (
            row["severity_top1_conf"] > config.high_confidence_threshold
        )
        high_conf_wrong.append(spec_wrong_high or sev_wrong_high)
    df["tax_high_confidence_wrong"] = high_conf_wrong

    # 6. Low-confidence uncertainty (highest confidence on either head is below thresholds)
    df["tax_low_confidence_uncertainty"] = (
        df["specialist_top1_conf"] < config.low_confidence_specialist_threshold
    ) | (df["severity_top1_conf"] < config.low_confidence_severity_threshold)

    return df


def generate_taxonomy_summary(df_classified: pd.DataFrame, config: Any) -> pd.DataFrame:
    """Summarize counts and percentages for each error taxonomy group.

    Args:
        df_classified: Classified predictions DataFrame containing tax_* columns.
        config: AnalysisConfig instance.
    """
    total_samples = len(df_classified)
    taxonomy_cols = [
        ("Wrong specialist, correct severity", "tax_wrong_spec_correct_sev"),
        ("Correct specialist, wrong severity", "tax_correct_spec_wrong_sev"),
        ("Both incorrect", "tax_both_incorrect"),
        ("Near-miss specialist", "tax_near_miss_specialist"),
        ("High-confidence wrong prediction", "tax_high_confidence_wrong"),
        ("Low-confidence uncertainty", "tax_low_confidence_uncertainty"),
    ]

    rows = []
    for label, col in taxonomy_cols:
        count = int(df_classified[col].sum())
        percentage = (count / total_samples) * 100 if total_samples > 0 else 0.0
        rows.append(
            {"Error Category": label, "Count": count, "Percentage (%)": percentage}
        )

    return pd.DataFrame(rows)
