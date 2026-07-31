"""Stage 9 – Global Shuffle.

Performs a deterministic, seed-controlled global shuffle of the merged
dataset.  Ensures reproducibility across runs with the same seed.

Writes:
  stage9_shuffled.parquet      – shuffled dataset
  stage9_shuffle_report.json   – shuffle metadata
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from reconstruction.config import ReconstructionConfig

logger = logging.getLogger(__name__)

STAGE_NAME = "stage9_shuffle"


def deterministic_shuffle(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Shuffle a DataFrame deterministically with the given seed.

    Args:
        df: Input DataFrame.
        seed: Random seed.

    Returns:
        Shuffled DataFrame with reset index.
    """
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def run(df: pd.DataFrame, cfg: ReconstructionConfig) -> pd.DataFrame:
    """Execute Stage 9: deterministic global shuffle."""
    out_dir = Path(cfg.output_directory)
    out_dir.mkdir(parents=True, exist_ok=True)

    output_path = out_dir / "stage9_shuffled.parquet"
    report_path = out_dir / "stage9_shuffle_report.json"

    if output_path.exists():
        logger.info("Stage 9 artifacts found, resuming from %s", output_path)
        return pd.read_parquet(output_path)

    shuffled = deterministic_shuffle(df, cfg.random_seed)

    shuffled.to_parquet(output_path, index=False)

    report = {
        "seed": cfg.random_seed,
        "total_rows": len(shuffled),
        "first_5_ids": shuffled["id"].head(5).tolist() if "id" in shuffled.columns else [],
        "last_5_ids": shuffled["id"].tail(5).tolist() if "id" in shuffled.columns else [],
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Stage 9 complete. Shuffled %d samples (seed=%d).", len(shuffled), cfg.random_seed)
    return shuffled
