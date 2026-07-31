"""Report Generator for Clinical Hard Negative Generation Engine."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from meditriage.multilingual.hard_negative.hard_negative_config import (
    HardNegativeConfig,
)

logger = logging.getLogger(__name__)


def generate_hard_negative_reports(
    df: pd.DataFrame,
    engine_stats: dict[str, Any],
    cfg: HardNegativeConfig,
) -> dict[str, Any]:
    """Generate all hard negative differential reports.

    Args:
        df: Expanded DataFrame containing hard negative samples.
        engine_stats: Engine statistics dictionary.
        cfg: Hard negative configuration.

    Returns:
        Master statistics report dictionary.
    """
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    master_report: dict[str, Any] = {}

    # 1. Hard Negative Report
    hn_report = {
        "total_source_records": engine_stats.get("total_source_records", 0),
        "total_hard_negatives_generated": engine_stats.get(
            "total_negatives_generated", 0
        ),
        "expansion_ratio": round(
            len(df) / max(engine_stats.get("total_source_records", 1), 1),
            2,
        ),
        "negatives_per_sample_setting": cfg.negatives_per_sample,
        "validation_passed": engine_stats.get("validation_passed", 0),
        "validation_failed": engine_stats.get("validation_failed", 0),
    }
    master_report["hard_negative"] = hn_report

    # 2. Confusion Pair Statistics
    diff_counts = engine_stats.get("differential_counts", {})
    confusion_stats = {
        "total_output_rows": len(df),
        "total_differential_confusion_pairs": len(diff_counts),
        "confusion_pair_breakdown": diff_counts,
        "random_seed": cfg.random_seed,
    }
    master_report["confusion_pair_statistics"] = confusion_stats

    # 3. Differential Coverage
    dept_dist = (
        df["department"].value_counts().to_dict() if "department" in df.columns else {}
    )
    coverage_report = {
        "target_department_distribution": dept_dist,
        "differentials_generated": list(diff_counts.keys()),
        "total_differentials_covered": len(diff_counts),
    }
    master_report["differential_coverage"] = coverage_report

    # Write output JSON files
    with open(out_dir / "hard_negative_report.json", "w", encoding="utf-8") as f:
        json.dump(hn_report, f, indent=2)

    with open(out_dir / "confusion_pair_statistics.json", "w", encoding="utf-8") as f:
        json.dump(confusion_stats, f, indent=2)

    with open(out_dir / "differential_coverage.json", "w", encoding="utf-8") as f:
        json.dump(coverage_report, f, indent=2)

    logger.info("Hard negative reports successfully generated in %s", out_dir)
    return master_report
