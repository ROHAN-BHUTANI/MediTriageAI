"""Stage 1 – Dataset Loading.

Loads the production dataset from parquet or CSV, validates required columns,
and writes a profile report as an intermediate artifact.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from reconstruction.config import ReconstructionConfig

logger = logging.getLogger(__name__)

STAGE_NAME = "stage1_load"


def load_dataset(cfg: ReconstructionConfig) -> pd.DataFrame:
    """Load the production dataset from parquet or CSV.

    Args:
        cfg: Reconstruction configuration.

    Returns:
        Raw pandas DataFrame.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If the file format is unsupported or required columns are missing.
    """
    path = Path(cfg.dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    logger.info("Loading dataset from %s", path)

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported dataset format: {path.suffix}. Use .parquet or .csv")

    # Validate required columns
    missing = set(cfg.required_columns) - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing)}. "
            f"Found: {df.columns.tolist()}"
        )

    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))
    return df


def generate_profile(df: pd.DataFrame) -> dict:
    """Generate a comprehensive profile of the loaded dataset.

    Args:
        df: Raw DataFrame.

    Returns:
        Profile dictionary.
    """
    profile: dict = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "columns": df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_counts": df.isnull().sum().to_dict(),
    }

    # Class distribution
    if "department" in df.columns:
        dept_counts = df["department"].value_counts(dropna=False).to_dict()
        # Convert NaN key to string
        profile["department_distribution"] = {
            str(k): int(v) for k, v in dept_counts.items()
        }

    if "triage_level" in df.columns:
        triage_counts = df["triage_level"].value_counts(dropna=False).to_dict()
        profile["triage_level_distribution"] = {
            str(k): int(v) for k, v in triage_counts.items()
        }

    if "language" in df.columns:
        lang_counts = df["language"].value_counts(dropna=False).to_dict()
        profile["language_distribution"] = {
            str(k): int(v) for k, v in lang_counts.items()
        }

    if "split" in df.columns:
        split_counts = df["split"].value_counts(dropna=False).to_dict()
        profile["split_distribution"] = {
            str(k): int(v) for k, v in split_counts.items()
        }

    if "raw_text" in df.columns:
        lengths = df["raw_text"].dropna().astype(str).str.len()
        profile["text_length_stats"] = {
            "mean": float(lengths.mean()),
            "std": float(lengths.std()),
            "min": int(lengths.min()),
            "max": int(lengths.max()),
            "median": float(lengths.median()),
        }

    return profile


def run(cfg: ReconstructionConfig) -> pd.DataFrame:
    """Execute Stage 1: load dataset and write profile artifact.

    Args:
        cfg: Reconstruction configuration.

    Returns:
        Raw DataFrame.
    """
    out_dir = Path(cfg.output_directory)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check for resume – if stage1 output already exists, reload it
    profile_path = out_dir / "stage1_dataset_profile.json"

    df = load_dataset(cfg)
    profile = generate_profile(df)

    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)

    logger.info("Stage 1 complete. Profile written to %s", profile_path)
    return df
