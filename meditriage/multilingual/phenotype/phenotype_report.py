"""Report Generator for Clinical Phenotype Augmentation Engine."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from meditriage.multilingual.phenotype.phenotype_config import PhenotypeConfig

logger = logging.getLogger(__name__)


def generate_phenotype_reports(
    df: pd.DataFrame,
    engine_stats: dict[str, Any],
    cfg: PhenotypeConfig,
) -> dict[str, Any]:
    """Generate all phenotype augmentation reports.

    Args:
        df: Expanded DataFrame with phenotype variants.
        engine_stats: Engine statistics dictionary.
        cfg: Phenotype configuration.

    Returns:
        Master statistics report dictionary.
    """
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    master_report: dict[str, Any] = {}

    # 1. Phenotype Generation Report
    gen_report = {
        "total_source_records": engine_stats.get("total_source_records", 0),
        "total_phenotype_variants_generated": engine_stats.get(
            "total_variants_generated", 0
        ),
        "augmentation_factor": round(
            len(df) / max(engine_stats.get("total_source_records", 1), 1),
            2,
        ),
        "enabled_specialties": cfg.enabled_specialties,
        "variants_per_sample_setting": cfg.variants_per_sample,
        "strict_consistency_checking": cfg.strict_consistency_checking,
    }
    master_report["generation"] = gen_report

    # 2. Phenotype Statistics
    phenotype_counts = engine_stats.get("phenotype_counts", {})
    stats_report = {
        "total_output_rows": len(df),
        "phenotypes_matched": len(phenotype_counts),
        "phenotype_breakdown": phenotype_counts,
        "average_variants_per_record": round(
            engine_stats.get("total_variants_generated", 0)
            / max(engine_stats.get("total_source_records", 1), 1),
            2,
        ),
        "random_seed": cfg.random_seed,
    }
    master_report["statistics"] = stats_report

    # 3. Phenotype Distribution
    dept_counts = (
        df["department"].value_counts().to_dict() if "department" in df.columns else {}
    )
    dist_report = {
        "department_distribution": dept_counts,
        "specialty_coverage": {
            spec: phenotype_counts.get(spec, 0) for spec in cfg.enabled_specialties
        },
    }
    master_report["distribution"] = dist_report

    # 4. Clinical Consistency Report
    val_passed = engine_stats.get("validation_passed", 0)
    val_failed = engine_stats.get("validation_failed", 0)
    total_val = val_passed + val_failed
    consistency_report = {
        "rule_engine_checks_passed": val_passed,
        "rule_engine_checks_failed": val_failed,
        "pass_rate_percentage": round((val_passed / max(total_val, 1)) * 100.0, 2),
        "impossible_combinations_prevented": val_failed,
    }
    master_report["clinical_consistency"] = consistency_report

    # Write output JSON files
    with open(out_dir / "phenotype_generation_report.json", "w", encoding="utf-8") as f:
        json.dump(gen_report, f, indent=2)

    with open(out_dir / "phenotype_statistics.json", "w", encoding="utf-8") as f:
        json.dump(stats_report, f, indent=2)

    with open(out_dir / "phenotype_distribution.json", "w", encoding="utf-8") as f:
        json.dump(dist_report, f, indent=2)

    with open(out_dir / "clinical_consistency_report.json", "w", encoding="utf-8") as f:
        json.dump(consistency_report, f, indent=2)

    logger.info("Phenotype augmentation reports successfully generated in %s", out_dir)
    return master_report
