"""Stage 6 – Multi-Axis Augmentation Engine.

For mid-tier classes (500 <= size < target), applies configurable
augmentation plugins to generate enough samples to reach target_class_size.

Writes:
  stage6_augmented.parquet          – augmented dataset
  stage6_augmentation_report.json   – plugin usage statistics
"""

from __future__ import annotations

import json
import logging
import random
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from reconstruction.augmentations import AugmentedSample
from reconstruction.augmentations.plugins import get_all_plugins
from reconstruction.config import ReconstructionConfig

logger = logging.getLogger(__name__)

STAGE_NAME = "stage6_augment"


def augment_class(
    class_df: pd.DataFrame,
    target_size: int,
    cfg: ReconstructionConfig,
) -> tuple[pd.DataFrame, dict]:
    """Augment a mid-tier class to reach target_size.

    Args:
        class_df: DataFrame for one department.
        target_size: Desired count.
        cfg: Reconstruction config.

    Returns:
        Tuple of (augmented DataFrame, report dict).
    """
    dept = class_df["department"].iloc[0]
    original_size = len(class_df)
    deficit = target_size - original_size

    if deficit <= 0:
        return class_df, {"department": dept, "action": "passthrough", "generated": 0}

    plugins = get_all_plugins()
    rng = random.Random(cfg.random_seed)
    augmented_rows = []
    plugin_usage: dict[str, int] = {}

    source_rows = class_df.to_dict("records")
    idx = 0

    while len(augmented_rows) < deficit:
        source = source_rows[idx % len(source_rows)]
        plugin = plugins[idx % len(plugins)]
        seed = cfg.random_seed + idx

        try:
            new_text = plugin.apply(source["raw_text"], seed=seed)
        except Exception:
            idx += 1
            continue

        if new_text and new_text != source["raw_text"]:
            row = dict(source)
            row["raw_text"] = new_text
            row["id"] = f"aug_{dept}_{idx}"
            row["dataset_source"] = f"augmented_{plugin.name}"
            row["_provenance_original_id"] = source.get("id", "")
            row["_provenance_plugin"] = plugin.name
            row["_provenance_seed"] = seed
            row["_provenance_timestamp"] = datetime.now(timezone.utc).isoformat()
            augmented_rows.append(row)
            plugin_usage[plugin.name] = plugin_usage.get(plugin.name, 0) + 1

        idx += 1
        if idx > deficit * 3:
            break  # Safety valve

    aug_df = pd.DataFrame(augmented_rows)
    result = pd.concat([class_df, aug_df], ignore_index=True)

    # Trim to exact target
    if len(result) > target_size:
        result = result.head(target_size)

    report = {
        "department": dept,
        "action": "augmented",
        "original_size": original_size,
        "generated": len(augmented_rows),
        "final_size": len(result),
        "plugin_usage": plugin_usage,
    }

    return result, report


def run(df: pd.DataFrame, cfg: ReconstructionConfig) -> pd.DataFrame:
    """Execute Stage 6: augment mid-tier classes."""
    out_dir = Path(cfg.output_directory)
    out_dir.mkdir(parents=True, exist_ok=True)

    output_path = out_dir / "stage6_augmented.parquet"
    report_path = out_dir / "stage6_augmentation_report.json"

    if output_path.exists():
        logger.info("Stage 6 artifacts found, resuming from %s", output_path)
        return pd.read_parquet(output_path)

    target = cfg.target_class_size
    all_parts = []
    all_reports = []

    for dept in sorted(df["department"].unique()):
        dept_df = df[df["department"] == dept].copy()
        size = len(dept_df)

        if size >= target:
            all_parts.append(dept_df)
            all_reports.append({"department": dept, "action": "already_at_target", "generated": 0})
        elif size >= cfg.augmentation_min_class_size:
            augmented, report = augment_class(dept_df, target, cfg)
            all_parts.append(augmented)
            all_reports.append(report)
            logger.info("  %s: %d -> %d (augmented)", dept, size, len(augmented))
        else:
            all_parts.append(dept_df)
            all_reports.append({"department": dept, "action": "needs_llm_generation", "size": size})

    result = pd.concat(all_parts, ignore_index=True)
    result.to_parquet(output_path, index=False)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"departments": all_reports}, f, indent=2)

    logger.info("Stage 6 complete. %d total samples.", len(result))
    return result
