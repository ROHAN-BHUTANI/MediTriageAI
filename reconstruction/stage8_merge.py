"""Stage 8 – Merge Engine.

Merges original, undersampled, augmented, and synthetic data into a
single unified dataset while preserving all provenance metadata.

Writes:
  stage8_merged.parquet        – merged dataset
  stage8_merge_report.json     – merge statistics
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from reconstruction.config import ReconstructionConfig

logger = logging.getLogger(__name__)

STAGE_NAME = "stage8_merge"


def merge_datasets(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Merge and compute statistics. The input is already the concatenation
    from previous stages; this stage normalises columns and computes stats.

    Args:
        df: Combined DataFrame from Stages 5-7.

    Returns:
        Tuple of (merged DataFrame, report dict).
    """
    report: dict = {
        "total_rows": len(df),
        "columns": df.columns.tolist(),
    }

    # Source breakdown
    if "dataset_source" in df.columns:
        source_counts = df["dataset_source"].value_counts().to_dict()
        report["source_breakdown"] = source_counts

        original = sum(v for k, v in source_counts.items() if not k.startswith(("augmented_", "synthetic_")))
        augmented = sum(v for k, v in source_counts.items() if k.startswith("augmented_"))
        synthetic = sum(v for k, v in source_counts.items() if k.startswith("synthetic_"))
        report["composition"] = {
            "original": original,
            "augmented": augmented,
            "synthetic": synthetic,
        }

    # Per-department counts
    if "department" in df.columns:
        report["department_counts"] = df["department"].value_counts().to_dict()

    # Language counts
    if "language" in df.columns:
        report["language_counts"] = df["language"].value_counts().to_dict()

    return df, report


def run(df: pd.DataFrame, cfg: ReconstructionConfig) -> pd.DataFrame:
    """Execute Stage 8: merge all data sources."""
    out_dir = Path(cfg.output_directory)
    out_dir.mkdir(parents=True, exist_ok=True)

    output_path = out_dir / "stage8_merged.parquet"
    report_path = out_dir / "stage8_merge_report.json"

    if output_path.exists():
        logger.info("Stage 8 artifacts found, resuming from %s", output_path)
        return pd.read_parquet(output_path)

    merged, report = merge_datasets(df)

    merged.to_parquet(output_path, index=False)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Stage 8 complete. Merged %d samples.", len(merged))
    return merged
