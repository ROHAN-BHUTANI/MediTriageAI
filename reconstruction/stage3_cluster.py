"""Stage 3 – Clinical Phenotype Clustering.

Groups samples within each department into clusters that represent distinct
clinical presentations.  Delegates encoding and clustering to a pluggable
ClusterBackend resolved through the backend factory.

Writes:
  stage3_clusters.parquet          – full dataset with cluster_id column
  stage3_cluster_statistics.json   – per-department cluster stats
"""

from __future__ import annotations

import gc
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
    """Heuristically determine a good cluster count for a department/batch.

    Uses sqrt(n) as a baseline, capped by max_clusters and ensuring each
    cluster would have at least min_cluster_size members on average.
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
    dept_name: str = "",
    start_cluster_id: int = 0,
) -> np.ndarray:
    """Cluster texts from a single department into clinical phenotype groups using batching.

    Args:
        texts: List of raw_text strings for one department.
        backend: The active ClusterBackend instance.
        cfg: Reconstruction configuration.
        dept_name: Name of department (for logging).
        start_cluster_id: Initial cluster ID offset for global uniqueness.

    Returns:
        NumPy array of cluster IDs (same length as texts).
    """
    n = len(texts)
    if n == 0:
        return np.array([], dtype=np.int32)
    if n == 1:
        return np.full(1, start_cluster_id, dtype=np.int32)

    batch_size = cfg.cluster_batch_size if cfg.cluster_batch_size > 0 else n
    n_batches = math.ceil(n / batch_size)

    labels_list = []
    current_offset = start_cluster_id

    for b_idx in range(n_batches):
        start_idx = b_idx * batch_size
        end_idx = min(start_idx + batch_size, n)
        batch_texts = texts[start_idx:end_idx]

        if dept_name:
            logger.info("Department %s", dept_name)
        logger.info("Batch %d/%d", b_idx + 1, n_batches)

        b_len = len(batch_texts)
        if b_len <= 1:
            batch_labels = np.full(b_len, current_offset, dtype=np.int32)
            current_offset += 1
        else:
            k = _determine_n_clusters(b_len, cfg.max_clusters_per_department, cfg.min_cluster_size)
            if k <= 1:
                batch_labels = np.full(b_len, current_offset, dtype=np.int32)
                current_offset += 1
            else:
                try:
                    backend.fit(batch_texts)
                    features = backend.encode(batch_texts)
                    raw_labels = backend.cluster(features, n_clusters=k, random_state=cfg.random_seed + b_idx)

                    batch_labels = (raw_labels + current_offset).astype(np.int32)
                    max_cluster_in_batch = int(np.max(raw_labels))
                    current_offset += max_cluster_in_batch + 1

                    del features, raw_labels
                except (ValueError, RuntimeError) as exc:
                    logger.warning("Clustering failed for batch (%s), assigning single cluster: %s", backend.name, exc)
                    batch_labels = np.full(b_len, current_offset, dtype=np.int32)
                    current_offset += 1

        labels_list.append(batch_labels)

        del batch_texts
        gc.collect()

    if labels_list:
        return np.concatenate(labels_list)
    return np.zeros(n, dtype=np.int32)


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
    """Execute Stage 3: cluster every department in batches and write artifacts.

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
    logger.info("Clustering %d departments (batch_size=%d)", len(departments), cfg.cluster_batch_size)

    global_offset = 0
    for dept in departments:
        mask = df["department"] == dept
        texts = df.loc[mask, "raw_text"].tolist()

        labels = cluster_department(
            texts=texts,
            backend=backend,
            cfg=cfg,
            dept_name=str(dept),
            start_cluster_id=global_offset,
        )
        df.loc[mask, "cluster_id"] = labels
        if len(labels) > 0:
            global_offset = int(np.max(labels)) + 1

        del texts, labels
        gc.collect()

    # Write artifacts
    df.to_parquet(clusters_path, index=False)

    stats = compute_cluster_statistics(df, backend.name)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    logger.info("Stage 3 complete. Artifacts written to %s", out_dir)
    return df

