"""Stage 3 – Clinical Phenotype Clustering.

Groups samples within each department into clusters that represent distinct
clinical presentations.  Delegates encoding and clustering to a pluggable
ClusterBackend resolved through the backend factory.

Writes:
  stage3_clusters.parquet          – full dataset with cluster_id column
  stage3_cluster_statistics.json   – per-department cluster stats
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

from reconstruction.backends import ClusterBackend
from reconstruction.backends.factory import create_backend
from reconstruction.config import ReconstructionConfig

logger = logging.getLogger(__name__)

STAGE_NAME = "stage3_cluster"


def _determine_n_clusters(n_samples: int, max_clusters: int, min_cluster_size: int) -> int:
    """Heuristically determine a good cluster count for a department.

    Uses sqrt(n) as a baseline, capped by max_clusters and ensuring each
    cluster would have at least min_cluster_size members on average.

    Args:
        n_samples: Number of samples in the department.
        max_clusters: Hard upper bound on cluster count.
        min_cluster_size: Minimum average cluster population.

    Returns:
        Number of clusters to use.
    """
    if n_samples < min_cluster_size * 2:
        return 1

    k = int(math.sqrt(n_samples))
    k = min(k, max_clusters)
    k = min(k, n_samples // min_cluster_size)
    k = max(k, 1)
    return k


def cluster_department(
    texts: list[str],
    backend: ClusterBackend,
    cfg: ReconstructionConfig,
) -> np.ndarray:
    """Cluster texts from a single department into clinical phenotype groups.

    Args:
        texts: List of raw_text strings for one department.
        backend: The active ClusterBackend instance.
        cfg: Reconstruction configuration.

    Returns:
        NumPy array of cluster IDs (same length as texts).
    """
    n = len(texts)
    if n <= 1:
        return np.zeros(n, dtype=np.int32)

    k = _determine_n_clusters(n, cfg.max_clusters_per_department, cfg.min_cluster_size)

    if k <= 1:
        return np.zeros(n, dtype=np.int32)

    # Delegate to backend
    try:
        backend.fit(texts)
        features = backend.encode(texts)
        labels = backend.cluster(features, n_clusters=k, random_state=cfg.random_seed)
    except (ValueError, RuntimeError) as exc:
        logger.warning("Clustering failed for department (%s), assigning single cluster: %s", backend.name, exc)
        return np.zeros(n, dtype=np.int32)

    return labels.astype(np.int32)


def compute_cluster_statistics(df: pd.DataFrame, backend_name: str) -> dict:
    """Compute per-department cluster statistics.

    Args:
        df: DataFrame with 'department' and 'cluster_id' columns.
        backend_name: Name of the backend used.

    Returns:
        Statistics dictionary.
    """
    stats: dict = {"backend": backend_name, "departments": {}}
    for dept, group in df.groupby("department"):
        cluster_counts = group["cluster_id"].value_counts().sort_index()
        stats["departments"][dept] = {
            "n_samples": len(group),
            "n_clusters": int(cluster_counts.nunique()),
            "cluster_sizes": {
                int(k): int(v) for k, v in cluster_counts.items()
            },
            "avg_cluster_size": float(cluster_counts.mean()),
            "min_cluster_size": int(cluster_counts.min()),
            "max_cluster_size": int(cluster_counts.max()),
        }
    return stats


def run(df: pd.DataFrame, cfg: ReconstructionConfig) -> pd.DataFrame:
    """Execute Stage 3: cluster every department and write artifacts.

    Args:
        df: Cleaned DataFrame from Stage 2.
        cfg: Reconstruction configuration.

    Returns:
        DataFrame with added 'cluster_id' column.
    """
    out_dir = Path(cfg.output_directory)
    out_dir.mkdir(parents=True, exist_ok=True)

    clusters_path = out_dir / "stage3_clusters.parquet"
    stats_path = out_dir / "stage3_cluster_statistics.json"

    # Resume support
    if clusters_path.exists():
        logger.info("Stage 3 artifacts found, resuming from %s", clusters_path)
        return pd.read_parquet(clusters_path)

    # Resolve the backend from config
    backend = create_backend(cfg)
    logger.info("Clustering backend: %s", backend.name)

    df = df.copy()
    df["cluster_id"] = -1

    departments = df["department"].unique()
    logger.info("Clustering %d departments", len(departments))

    for dept in departments:
        mask = df["department"] == dept
        texts = df.loc[mask, "raw_text"].tolist()
        logger.info("  %s: %d samples", dept, len(texts))

        labels = cluster_department(texts, backend, cfg)
        df.loc[mask, "cluster_id"] = labels

    # Write artifacts
    df.to_parquet(clusters_path, index=False)

    stats = compute_cluster_statistics(df, backend.name)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    logger.info("Stage 3 complete. Artifacts written to %s", out_dir)
    return df
