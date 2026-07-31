"""Stage 2 – Cleaning.

Removes invalid rows, normalizes unicode/whitespace, and writes a cleaning
report as an intermediate artifact.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from pathlib import Path

import pandas as pd

from reconstruction.config import ReconstructionConfig

logger = logging.getLogger(__name__)

STAGE_NAME = "stage2_clean"


def normalize_text(text: str) -> str:
    """Normalize unicode and whitespace in a text string.

    Args:
        text: Raw input string.

    Returns:
        Cleaned string.
    """
    # Unicode NFC normalization
    text = unicodedata.normalize("NFC", text)
    # Replace various unicode whitespace with regular space
    text = re.sub(r"[\u00a0\u2000-\u200b\u202f\u205f\u3000\ufeff]", " ", text)
    # Collapse multiple whitespace
    text = re.sub(r"\s+", " ", text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text


def is_valid_text(text: str, min_length: int, max_length: int) -> bool:
    """Check if a text string meets validity criteria.

    Args:
        text: Normalized text.
        min_length: Minimum character count.
        max_length: Maximum character count.

    Returns:
        True if valid.
    """
    if not text:
        return False
    length = len(text)
    if length < min_length or length > max_length:
        return False
    # Reject strings that are only special characters / numbers
    if not re.search(r"[a-zA-Z\u0900-\u097F]", text):
        return False
    return True


def clean_dataset(df: pd.DataFrame, cfg: ReconstructionConfig) -> tuple[pd.DataFrame, dict]:
    """Apply cleaning rules and return the cleaned DataFrame plus a report.

    Args:
        df: Input DataFrame.
        cfg: Reconstruction configuration.

    Returns:
        Tuple of (cleaned DataFrame, cleaning report dict).
    """
    initial_rows = len(df)
    report: dict = {"initial_rows": initial_rows, "dropped": {}}

    # Drop rows missing raw_text
    mask_no_text = df["raw_text"].isna()
    count_no_text = int(mask_no_text.sum())
    df = df[~mask_no_text].copy()
    report["dropped"]["missing_raw_text"] = count_no_text

    # Drop rows missing department
    mask_no_dept = df["department"].isna()
    count_no_dept = int(mask_no_dept.sum())
    df = df[~mask_no_dept].copy()
    report["dropped"]["missing_department"] = count_no_dept

    # Normalize text
    df["raw_text"] = df["raw_text"].astype(str).apply(normalize_text)

    # Filter by text validity
    validity_mask = df["raw_text"].apply(
        lambda t: is_valid_text(t, cfg.min_text_length, cfg.max_text_length)
    )
    count_invalid = int((~validity_mask).sum())
    df = df[validity_mask].copy()
    report["dropped"]["invalid_text"] = count_invalid

    # Remove exact duplicate texts within the same department
    pre_dedup = len(df)
    df = df.drop_duplicates(subset=["raw_text", "department"], keep="first")
    count_dupes = pre_dedup - len(df)
    report["dropped"]["duplicate_text_per_dept"] = count_dupes

    final_rows = len(df)
    report["final_rows"] = final_rows
    report["total_dropped"] = initial_rows - final_rows

    # Post-cleaning class distribution
    report["department_distribution"] = (
        df["department"].value_counts().to_dict()
    )

    logger.info(
        "Cleaning complete: %d -> %d rows (dropped %d)",
        initial_rows, final_rows, initial_rows - final_rows,
    )
    return df, report


def run(df: pd.DataFrame, cfg: ReconstructionConfig) -> pd.DataFrame:
    """Execute Stage 2: clean dataset and write report artifact.

    Args:
        df: Raw DataFrame from Stage 1.
        cfg: Reconstruction configuration.

    Returns:
        Cleaned DataFrame.
    """
    out_dir = Path(cfg.output_directory)
    out_dir.mkdir(parents=True, exist_ok=True)

    report_path = out_dir / "stage2_cleaning_report.json"

    df_clean, report = clean_dataset(df, cfg)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Stage 2 complete. Report written to %s", report_path)
    return df_clean
