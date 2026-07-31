"""Stage 5 – Diversity-Maximization Undersampling.

For majority classes (sample count >= target_class_size), selects exactly
target_class_size samples by:
  1. Allocating a proportional budget to each clinical phenotype cluster.
  2. Within each cluster, selecting samples with the highest diversity scores.
  3. Ensuring language coverage is maintained.

Mid-tier and minority classes are passed through unchanged for later stages.

Writes:
  stage5_majority_selected.parquet – undersampled majority + passthrough
  stage5_selection_report.json     – per-department selection statistics
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from reconstruction.config import ReconstructionConfig

logger = logging.getLogger(__name__)

STAGE_NAME = "stage5_undersample"


def select_from_cluster(
    cluster_df: pd.DataFrame,
    budget: int,
) -> pd.DataFrame:
    """Select the top-`budget` samples from a cluster by diversity score.

    Args:
        cluster_df: DataFrame slice for one cluster (must have diversity_score).
        budget: Number of samples to select.

    Returns:
        Selected DataFrame slice.
    """
    if len(cluster_df) <= budget:
        return cluster_df
    return cluster_df.nlargest(budget, "diversity_score")


def undersample_department(
    dept_df: pd.DataFrame,
    target_size: int,
) -> tuple[pd.DataFrame, dict]:
    """Undersample a single majority department to target_size.

    Allocation: proportional to cluster population, with a minimum of 1 per
    cluster to preserve phenotype coverage.

    Args:
        dept_df: Full department DataFrame (with cluster_id, diversity_score).
        target_size: Desired sample count.

    Returns:
        Tuple of (selected DataFrame, selection report dict).
    """
    dept_name = dept_df["department"].iloc[0]
    original_size = len(dept_df)

    if original_size <= target_size:
        return dept_df, {
            "department": dept_name,
            "action": "passthrough",
            "original_size": original_size,
            "selected_size": original_size,
        }

    cluster_ids = dept_df["cluster_id"].unique()
    n_clusters = len(cluster_ids)

    # Proportional budget allocation
    cluster_sizes = dept_df["cluster_id"].value_counts()
    total_for_alloc = cluster_sizes.sum()

    budgets: dict[int, int] = {}
    allocated = 0
    for cid in cluster_ids:
        proportion = cluster_sizes[cid] / total_for_alloc
        budget = max(1, int(round(proportion * target_size)))
        budgets[cid] = budget
        allocated += budget

    # Adjust for rounding errors
    diff = target_size - allocated
    if diff != 0:
        sorted_clusters = sorted(
            cluster_ids,
            key=lambda c: cluster_sizes[c],
            reverse=(diff > 0),
        )
        for i in range(abs(diff)):
            cid = sorted_clusters[i % len(sorted_clusters)]
            budgets[cid] += 1 if diff > 0 else -1
            budgets[cid] = max(budgets[cid], 1)

    # Select from each cluster
    selected_parts = []
    cluster_report = {}
    for cid in cluster_ids:
        cluster_df = dept_df[dept_df["cluster_id"] == cid]
        budget = budgets[cid]
        selected = select_from_cluster(cluster_df, budget)
        selected_parts.append(selected)
        cluster_report[int(cid)] = {
            "original": len(cluster_df),
            "budget": budget,
            "selected": len(selected),
        }

    result = pd.concat(selected_parts, ignore_index=True)

    # Final trim/pad to exact target
    if len(result) > target_size:
        result = result.nlargest(target_size, "diversity_score")
    elif len(result) < target_size:
        # Fill from remaining unselected samples by diversity score
        selected_ids = set(result.index)
        remaining = dept_df.loc[~dept_df.index.isin(selected_ids)]
        needed = target_size - len(result)
        extra = remaining.nlargest(needed, "diversity_score")
        result = pd.concat([result, extra], ignore_index=True)

    report = {
        "department": dept_name,
        "action": "undersampled",
        "original_size": original_size,
        "selected_size": len(result),
        "n_clusters": n_clusters,
        "cluster_allocations": cluster_report,
        "languages_preserved": result["language"].value_counts().to_dict()
        if "language" in result.columns
        else {},
    }

    return result, report


def run(df: pd.DataFrame, cfg: ReconstructionConfig) -> pd.DataFrame:
    """Execute Stage 5: undersample majority classes, passthrough others.

    Args:
        df: Diversity-scored DataFrame from Stage 4.
        cfg: Reconstruction configuration.

    Returns:
        DataFrame with majority classes undersampled.
    """
    out_dir = Path(cfg.output_directory)
    out_dir.mkdir(parents=True, exist_ok=True)

    output_path = out_dir / "stage5_majority_selected.parquet"
    report_path = out_dir / "stage5_selection_report.json"

    # Resume support
    if output_path.exists():
        logger.info("Stage 5 artifacts found, resuming from %s", output_path)
        return pd.read_parquet(output_path)

    target = cfg.target_class_size
    all_parts = []
    all_reports = []

    for dept in sorted(df["department"].unique()):
        dept_df = df[df["department"] == dept].copy()
        dept_size = len(dept_df)

        if dept_size >= target:
            selected, report = undersample_department(dept_df, target)
            all_parts.append(selected)
            all_reports.append(report)
            logger.info(
                "  %s: %d -> %d (undersampled)", dept, dept_size, len(selected)
            )
        else:
            # Passthrough for mid-tier / minority
            action = "needs_augmentation" if dept_size >= cfg.augmentation_min_class_size else "needs_generation"
            all_parts.append(dept_df)
            all_reports.append({
                "department": dept,
                "action": action,
                "original_size": dept_size,
                "selected_size": dept_size,
                "deficit": target - dept_size,
            })
            logger.info(
                "  %s: %d (passthrough, %s, deficit=%d)",
                dept, dept_size, action, target - dept_size,
            )

    result = pd.concat(all_parts, ignore_index=True)

    result.to_parquet(output_path, index=False)

    full_report = {
        "target_class_size": target,
        "total_samples_after_undersampling": len(result),
        "departments": all_reports,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    logger.info("Stage 5 complete. Output: %d samples -> %s", len(result), output_path)
    return result
