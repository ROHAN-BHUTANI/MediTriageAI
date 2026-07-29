"""Classification, Top-k, and Bootstrap metrics for the MediTriageAI analysis framework."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report as sk_classification_report
from sklearn.metrics import f1_score


def compute_prediction_entropy(probs: np.ndarray) -> np.ndarray:
    """Compute Shannon entropy (base 2) for each probability vector.
    
    H(P) = - sum(p_i * log2(p_i))
    """
    # Clip to avoid log(0)
    probs_clipped = np.clip(probs, 1e-15, 1.0)
    return -np.sum(probs_clipped * np.log2(probs_clipped), axis=1)


def compute_top_k_accuracy(probs: np.ndarray, y_true: np.ndarray, k: int) -> float:
    """Compute Top-k accuracy.
    
    Checks if the true class index is in the top-k highest predicted probabilities.
    """
    if len(probs) == 0:
        return 0.0
    # Argsort sorts ascending, so we take the last k elements
    top_k_indices = np.argsort(probs, axis=1)[:, -k:]
    # Check if y_true is in the top k predicted indices for each sample using vectorized numpy comparison
    matches = (top_k_indices == y_true[:, np.newaxis]).any(axis=1)
    return float(np.mean(matches))



def compute_overall_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute overall accuracy, Macro F1, and Weighted F1."""
    if len(y_true) == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0}
    from src.metrics import compute_macro_f1
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(compute_macro_f1(y_true, y_pred, "specialist")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def compute_per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> pd.DataFrame:
    """Compute per-class precision, recall, and F1-score.
    
    Returns a DataFrame with columns: Class, Precision, Recall, F1, Support.
    """
    if len(y_true) == 0:
        return pd.DataFrame(columns=["Class", "Precision", "Recall", "F1", "Support"])
        
    report = sk_classification_report(
        y_true,
        y_pred,
        labels=classes,
        output_dict=True,
        zero_division=0
    )
    
    rows = []
    for cls in classes:
        cls_data = report.get(cls, {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 0})
        rows.append({
            "Class": cls,
            "Precision": float(cls_data["precision"]),
            "Recall": float(cls_data["recall"]),
            "F1": float(cls_data["f1-score"]),
            "Support": int(cls_data["support"])
        })
        
    return pd.DataFrame(rows)


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_resamples: int = 1000,
    seed: int = 42
) -> tuple[float, float, float]:
    """Calculate the mean and 95% confidence interval (CI) using bootstrapping.
    
    Returns:
        A tuple of (mean, 95% CI lower bound, 95% CI upper bound).
    """
    if len(y_true) == 0:
        return 0.0, 0.0, 0.0
        
    rng = np.random.default_rng(seed)
    n = len(y_true)
    bootstrap_scores = []
    
    for _ in range(n_resamples):
        indices = rng.choice(n, size=n, replace=True)
        bootstrap_scores.append(metric_fn(y_true[indices], y_pred[indices]))
        
    bootstrap_scores = np.sort(bootstrap_scores)
    ci_lower = float(np.percentile(bootstrap_scores, 2.5))
    ci_upper = float(np.percentile(bootstrap_scores, 97.5))
    mean_val = float(np.mean(bootstrap_scores))
    
    return mean_val, ci_lower, ci_upper


def add_confidence_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute and append confidence metrics to predictions DataFrame.
    
    Appends:
        - specialist_entropy
        - specialist_top1_conf
        - specialist_top2_conf
        - specialist_margin
        - severity_entropy
        - severity_top1_conf
        - severity_top2_conf
        - severity_margin
    """
    df = df.copy()
    
    # 1. Specialist confidence metrics
    spec_probs = np.vstack(df["specialist_probabilities"].values)
    df["specialist_entropy"] = compute_prediction_entropy(spec_probs)
    
    # Top-1 and Top-2 Specialist Confidences
    spec_sorted = np.sort(spec_probs, axis=1)
    df["specialist_top1_conf"] = spec_sorted[:, -1]
    df["specialist_top2_conf"] = spec_sorted[:, -2]
    df["specialist_margin"] = df["specialist_top1_conf"] - df["specialist_top2_conf"]
    
    # 2. Severity confidence metrics
    sev_probs = np.vstack(df["severity_probabilities"].values)
    df["severity_entropy"] = compute_prediction_entropy(sev_probs)
    
    # Top-1 and Top-2 Severity Confidences
    sev_sorted = np.sort(sev_probs, axis=1)
    df["severity_top1_conf"] = sev_sorted[:, -1]
    df["severity_top2_conf"] = sev_sorted[:, -2]
    df["severity_margin"] = df["severity_top1_conf"] - df["severity_top2_conf"]
    
    return df


def fast_macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Fast vectorized macro F1 score computation in pure NumPy."""
    tp = np.zeros(n_classes)
    fp = np.zeros(n_classes)
    fn = np.zeros(n_classes)
    for c in range(n_classes):
        tp[c] = np.sum((y_true == c) & (y_pred == c))
        fp[c] = np.sum((y_true != c) & (y_pred == c))
        fn[c] = np.sum((y_true == c) & (y_pred != c))
    
    prec = np.zeros(n_classes)
    rec = np.zeros(n_classes)
    
    tp_fp = tp + fp
    tp_fn = tp + fn
    
    prec_mask = tp_fp > 0
    rec_mask = tp_fn > 0
    
    prec[prec_mask] = tp[prec_mask] / tp_fp[prec_mask]
    rec[rec_mask] = tp[rec_mask] / tp_fn[rec_mask]
    
    prec_rec = prec + rec
    f1 = np.zeros(n_classes)
    f1_mask = prec_rec > 0
    f1[f1_mask] = 2 * prec[f1_mask] * rec[f1_mask] / prec_rec[f1_mask]
    
    return float(np.mean(f1))


def fast_weighted_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Fast vectorized weighted F1 score computation in pure NumPy."""
    tp = np.zeros(n_classes)
    fp = np.zeros(n_classes)
    fn = np.zeros(n_classes)
    for c in range(n_classes):
        tp[c] = np.sum((y_true == c) & (y_pred == c))
        fp[c] = np.sum((y_true != c) & (y_pred == c))
        fn[c] = np.sum((y_true == c) & (y_pred != c))
    
    prec = np.zeros(n_classes)
    rec = np.zeros(n_classes)
    
    tp_fp = tp + fp
    tp_fn = tp + fn
    
    prec_mask = tp_fp > 0
    rec_mask = tp_fn > 0
    
    prec[prec_mask] = tp[prec_mask] / tp_fp[prec_mask]
    rec[rec_mask] = tp[rec_mask] / tp_fn[rec_mask]
    
    prec_rec = prec + rec
    f1 = np.zeros(n_classes)
    f1_mask = prec_rec > 0
    f1[f1_mask] = 2 * prec[f1_mask] * rec[f1_mask] / prec_rec[f1_mask]
    
    support = tp_fn
    total_support = np.sum(support)
    if total_support == 0:
        return 0.0
    return float(np.sum(f1 * support) / total_support)

