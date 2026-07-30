"""Utility functions for seeding, logging, hashing, and statistical tests in the analysis framework."""

from __future__ import annotations

import hashlib
import logging
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seeds across random, numpy, and PyTorch for reproducibility."""
    random.seed(seed)
    os_seed = str(seed)
    sys.modules["os"].environ["PYTHONHASHSEED"] = os_seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Enable deterministic behavior in CuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_logger(log_file: Path) -> logging.Logger:
    """Configure structured file and console logging."""
    logger = logging.getLogger("MediTriageAnalysis")
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if logger is already set up
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s [%(name)s:%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def compute_sha256(file_path: Path) -> str:
    """Compute the SHA256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read in blocks to handle large files efficiently
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def compute_mcnemar_test(
    y_true: np.ndarray, y_pred1: np.ndarray, y_pred2: np.ndarray
) -> dict[str, Any]:
    """Perform McNemar's test to compare prediction accuracy of two models.

    Args:
        y_true: 1D array of ground truth labels.
        y_pred1: 1D array of predictions from model 1.
        y_pred2: 1D array of predictions from model 2.

    Returns:
        A dictionary with chi-square statistic, p-value, and the contingency counts.
    """
    from scipy.stats import chi2

    # Boolean arrays indicating correctness
    m1_correct = y_pred1 == y_true
    m2_correct = y_pred2 == y_true

    # Contingency matrix elements
    # b: Model 1 correct, Model 2 incorrect
    # c: Model 1 incorrect, Model 2 correct
    b = int(np.sum(m1_correct & ~m2_correct))
    c = int(np.sum(~m1_correct & m2_correct))

    # Compute chi-square statistic with Edwards continuity correction
    if b + c > 0:
        stat = float(((abs(b - c) - 1.0) ** 2) / (b + c))
        # 1 degree of freedom
        p_value = float(1.0 - chi2.cdf(stat, df=1))
    else:
        stat = 0.0
        p_value = 1.0

    return {
        "statistic": stat,
        "p_value": p_value,
        "model1_only_correct": b,
        "model2_only_correct": c,
        "total_disagreements": b + c,
        "significant": p_value < 0.05,
    }


def df_to_markdown(df: pd.DataFrame) -> str:
    """Convert a pandas DataFrame to a markdown table string.

    This replaces pandas to_markdown which requires the external 'tabulate' library.
    """
    if df.empty:
        return ""
    headers = [str(c) for c in df.columns]

    # Calculate column widths
    widths = []
    for c in df.columns:
        col_vals = [str(x) for x in df[c].values]
        val_len = max(len(x) for x in col_vals) if col_vals else 0
        widths.append(max(val_len, len(str(c))))

    header_line = "| " + " | ".join(f"{h:<{w}}" for h, w in zip(headers, widths)) + " |"
    sep_line = "| " + " | ".join("-" * w for w in widths) + " |"

    data_lines = []
    for _, row in df.iterrows():
        row_str = [str(x) for x in row.values]
        line = (
            "| " + " | ".join(f"{val:<{w}}" for val, w in zip(row_str, widths)) + " |"
        )
        data_lines.append(line)

    return "\n".join([header_line, sep_line] + data_lines)
