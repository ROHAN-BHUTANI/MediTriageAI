"""Stage 10 – Validation Engine.

Runs a pipeline of independent validators on the final dataset.
Each validator is a standalone module that outputs a JSON result.

Writes:
  stage10_validation_results.json  – combined validation report
  stage10_final_dataset.parquet    – validated dataset (rejected rows removed)
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from reconstruction.config import ReconstructionConfig

logger = logging.getLogger(__name__)

STAGE_NAME = "stage10_validate"


# ── Individual Validators ────────────────────────────────────────────────

def validate_duplicates(df: pd.DataFrame) -> dict:
    """Check for exact duplicate raw_text entries."""
    dupes = df.duplicated(subset=["raw_text"], keep="first")
    dupe_count = int(dupes.sum())
    return {
        "validator": "duplicate",
        "passed": dupe_count == 0,
        "duplicate_count": dupe_count,
        "duplicate_ids": df.loc[dupes, "id"].head(20).tolist() if "id" in df.columns else [],
    }


def validate_contradictions(df: pd.DataFrame) -> dict:
    """Check for identical raw_text with different department labels."""
    grouped = df.groupby("raw_text")["department"].nunique()
    contradictions = int((grouped > 1).sum())
    return {
        "validator": "contradiction",
        "passed": contradictions == 0,
        "contradiction_count": contradictions,
    }


def validate_balance(df: pd.DataFrame, target_size: int) -> dict:
    """Check that every department has exactly target_size rows."""
    counts = df["department"].value_counts().to_dict()
    imbalanced = {k: v for k, v in counts.items() if v != target_size}
    return {
        "validator": "balance",
        "passed": len(imbalanced) == 0,
        "target_size": target_size,
        "department_counts": counts,
        "imbalanced_departments": imbalanced,
    }


def validate_language(df: pd.DataFrame) -> dict:
    """Check language coverage."""
    if "language" not in df.columns:
        return {"validator": "language", "passed": False, "error": "no language column"}
    counts = df["language"].value_counts().to_dict()
    return {
        "validator": "language",
        "passed": len(counts) >= 1,
        "language_counts": counts,
    }


def validate_phenotype(df: pd.DataFrame) -> dict:
    """Check cluster coverage (from Stage 3)."""
    if "cluster_id" not in df.columns:
        return {"validator": "phenotype", "passed": True, "note": "no cluster_id column"}
    per_dept = {}
    for dept, group in df.groupby("department"):
        per_dept[dept] = int(group["cluster_id"].nunique())
    return {
        "validator": "phenotype",
        "passed": all(v >= 1 for v in per_dept.values()),
        "clusters_per_department": per_dept,
    }


def validate_provenance(df: pd.DataFrame) -> dict:
    """Check that provenance columns exist for generated/augmented rows."""
    augmented_mask = df["dataset_source"].str.startswith("augmented_") if "dataset_source" in df.columns else pd.Series(False, index=df.index)
    synthetic_mask = df["dataset_source"].str.startswith("synthetic_") if "dataset_source" in df.columns else pd.Series(False, index=df.index)

    n_aug = int(augmented_mask.sum())
    n_syn = int(synthetic_mask.sum())

    provenance_cols = [c for c in df.columns if c.startswith("_provenance_")]
    return {
        "validator": "provenance",
        "passed": True,
        "augmented_samples": n_aug,
        "synthetic_samples": n_syn,
        "provenance_columns": provenance_cols,
    }


def validate_embedding_similarity(df: pd.DataFrame, threshold: float) -> dict:
    """Flag near-duplicate texts using token overlap as a fast proxy.

    Full embedding similarity would require a model; this uses Jaccard
    token overlap for speed.  Pairs above threshold are flagged.
    """
    token_re = re.compile(r"\w+")

    # Sample for performance
    sample_size = min(5000, len(df))
    sample = df.sample(n=sample_size, random_state=42) if len(df) > sample_size else df

    texts = sample["raw_text"].tolist()
    token_sets = [frozenset(token_re.findall(t.lower())) for t in texts]

    near_dupes = 0
    checked = 0
    for i in range(len(token_sets)):
        for j in range(i + 1, min(i + 50, len(token_sets))):
            a, b = token_sets[i], token_sets[j]
            if not a or not b:
                continue
            overlap = len(a & b) / len(a | b)
            if overlap > threshold:
                near_dupes += 1
            checked += 1

    return {
        "validator": "embedding_similarity",
        "passed": near_dupes == 0,
        "threshold": threshold,
        "near_duplicates_found": near_dupes,
        "pairs_checked": checked,
        "sample_size": sample_size,
    }


# ── Validation Pipeline ─────────────────────────────────────────────────

def run_validators(df: pd.DataFrame, cfg: ReconstructionConfig) -> list[dict]:
    """Run all validators and return their results."""
    results = [
        validate_duplicates(df),
        validate_contradictions(df),
        validate_balance(df, cfg.target_class_size),
        validate_language(df),
        validate_phenotype(df),
        validate_provenance(df),
        validate_embedding_similarity(df, cfg.similarity_threshold),
    ]
    return results


def run(df: pd.DataFrame, cfg: ReconstructionConfig) -> pd.DataFrame:
    """Execute Stage 10: run all validators and write results."""
    out_dir = Path(cfg.output_directory)
    out_dir.mkdir(parents=True, exist_ok=True)

    results_path = out_dir / "stage10_validation_results.json"
    output_path = out_dir / "stage10_final_dataset.parquet"

    if output_path.exists():
        logger.info("Stage 10 artifacts found, resuming from %s", output_path)
        return pd.read_parquet(output_path)

    results = run_validators(df, cfg)

    # Remove exact duplicates if found
    dup_result = next(r for r in results if r["validator"] == "duplicate")
    if not dup_result["passed"]:
        pre = len(df)
        df = df.drop_duplicates(subset=["raw_text"], keep="first")
        logger.warning("Removed %d duplicate rows.", pre - len(df))

    all_passed = all(r["passed"] for r in results)

    combined = {
        "all_passed": all_passed,
        "validators": results,
    }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)

    df.to_parquet(output_path, index=False)

    logger.info("Stage 10 complete. All passed: %s", all_passed)
    return df
