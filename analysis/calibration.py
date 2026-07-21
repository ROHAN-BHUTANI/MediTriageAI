"""Calibration evaluation metrics for the MediTriageAI analysis framework."""

from __future__ import annotations

import numpy as np


def compute_ece_mce(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> tuple[float, float]:
    """Compute Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
    
    Args:
        probs: (N, C) predicted probability distributions.
        labels: (N,) true class indices.
        n_bins: Number of confidence bins.
        
    Returns:
        A tuple of (ece, mce).
    """
    preds = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)
    accuracies = (preds == labels)
    
    ece = 0.0
    mce = 0.0
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Handle the upper boundary on the last bin
        if i == n_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
            
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            bin_error = np.abs(avg_confidence_in_bin - accuracy_in_bin)
            
            ece += prop_in_bin * bin_error
            mce = max(mce, bin_error)
            
    return float(ece), float(mce)


def compute_nll(probs: np.ndarray, labels: np.ndarray) -> float:
    """Compute Negative Log-Likelihood (NLL) for a set of predictions.
    
    Args:
        probs: (N, C) predicted probability distributions.
        labels: (N,) true class indices.
    """
    n = len(labels)
    if n == 0:
        return 0.0
    true_probs = probs[np.arange(n), labels]
    return float(-np.mean(np.log(np.clip(true_probs, 1e-15, 1.0))))


def compute_brier_score(probs: np.ndarray, labels: np.ndarray, num_classes: int) -> float:
    """Compute the multi-class Brier score.
    
    Args:
        probs: (N, C) predicted probability distributions.
        labels: (N,) true class indices.
        num_classes: Number of target classes.
    """
    n = len(labels)
    if n == 0:
        return 0.0
    one_hot = np.zeros((n, num_classes))
    one_hot[np.arange(n), labels] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def get_reliability_curve_data(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> dict[str, list[float]]:
    """Compute reliability curve statistics for plotting.
    
    Returns:
        A dictionary with keys 'bin_confidences', 'bin_accuracies', and 'bin_counts'.
    """
    preds = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)
    accuracies = (preds == labels)
    
    bin_confidences = []
    bin_accuracies = []
    bin_counts = []
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        if i == n_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
            
        count = int(np.sum(in_bin))
        bin_counts.append(count)
        
        if count > 0:
            bin_confidences.append(float(np.mean(confidences[in_bin])))
            bin_accuracies.append(float(np.mean(accuracies[in_bin])))
        else:
            # If no samples fall in the bin, represent as midpoints
            bin_confidences.append(float((bin_lower + bin_upper) / 2))
            bin_accuracies.append(0.0)
            
    return {
        "bin_confidences": bin_confidences,
        "bin_accuracies": bin_accuracies,
        "bin_counts": bin_counts,
    }
