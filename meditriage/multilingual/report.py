"""Multilingual Dataset Expansion Audit and Reporting Generator."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from meditriage.multilingual.config import MultilingualConfig

logger = logging.getLogger(__name__)


def generate_multilingual_reports(
    df: pd.DataFrame,
    translator_stats: dict[str, Any],
    cfg: MultilingualConfig,
) -> dict[str, Any]:
    """Generate all audit and validation reports for multilingual expansion.

    Args:
        df: Expanded DataFrame.
        translator_stats: Statistics dict from MultilingualTranslator.
        cfg: Multilingual configuration.

    Returns:
        Master audit dictionary.
    """
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    master_report: dict[str, Any] = {}

    # 1. Language counts and distribution
    lang_counts = (
        df["language"].value_counts().to_dict() if "language" in df.columns else {}
    )
    total_rows = len(df)
    lang_dist = {
        lang: {
            "count": count,
            "percentage": round(count / max(total_rows, 1) * 100, 2),
        }
        for lang, count in lang_counts.items()
    }
    master_report["language_distribution"] = lang_dist

    # 2. Multilingual expansion summary
    expansion_report = {
        "total_input_rows": translator_stats.get("total_input_rows", 0),
        "total_output_rows": total_rows,
        "expansion_factor": round(
            total_rows / max(translator_stats.get("total_input_rows", 1), 1), 2
        ),
        "target_languages": cfg.target_languages,
        "provider_used": cfg.provider,
        "model_name": cfg.model_name,
        "cache_hits": translator_stats.get("cache_hits", 0),
        "cache_misses": translator_stats.get("cache_misses", 0),
        "elapsed_seconds": translator_stats.get("elapsed_seconds", 0),
    }
    master_report["expansion_summary"] = expansion_report

    # 3. Quality & validation summary
    quality_report = {
        "validation_passed": translator_stats.get("validation_passed", 0),
        "validation_failed": translator_stats.get("validation_failed", 0),
        "strict_validation_enabled": cfg.strict_validation,
        "pass_rate_percentage": round(
            translator_stats.get("validation_passed", 0)
            / max(
                translator_stats.get("validation_passed", 0)
                + translator_stats.get("validation_failed", 0),
                1,
            )
            * 100,
            2,
        ),
    }
    master_report["quality_summary"] = quality_report

    # Write JSON files
    with open(
        out_dir / "multilingual_expansion_report.json", "w", encoding="utf-8"
    ) as f:
        json.dump(expansion_report, f, indent=2)

    with open(
        out_dir / "multilingual_language_distribution.json", "w", encoding="utf-8"
    ) as f:
        json.dump(lang_dist, f, indent=2)

    with open(out_dir / "multilingual_quality_report.json", "w", encoding="utf-8") as f:
        json.dump(quality_report, f, indent=2)

    with open(
        out_dir / "multilingual_validation_report.json", "w", encoding="utf-8"
    ) as f:
        json.dump(master_report, f, indent=2)

    # Write Markdown distribution report
    md_path = out_dir / "multilingual_language_distribution.md"
    md_content = _build_markdown_report(expansion_report, lang_dist, quality_report)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    logger.info("Multilingual audit reports generated in %s", out_dir)
    return master_report


def _build_markdown_report(expansion: dict, lang_dist: dict, quality: dict) -> str:
    lines = ["# Multilingual Dataset Expansion Report\n"]

    lines.append("## Executive Summary\n")
    lines.append(f"- **Total Input Rows**: {expansion.get('total_input_rows')}")
    lines.append(f"- **Total Output Rows**: {expansion.get('total_output_rows')}")
    lines.append(f"- **Expansion Factor**: {expansion.get('expansion_factor')}x")
    lines.append(
        f"- **Provider**: `{expansion.get('provider_used')}` ({expansion.get('model_name')})"
    )
    lines.append(f"- **Elapsed Time**: {expansion.get('elapsed_seconds')}s\n")

    lines.append("## Language Distribution\n")
    lines.append("| Language Code | Description | Count | Percentage |")
    lines.append("|---------------|-------------|-------|------------|")

    names = {
        "en": "English (Original)",
        "hi": "Hindi (Devanagari)",
        "hi-Latn": "Roman Hindi (Latin Script)",
        "hi-en": "Natural Hinglish",
        "en-hi": "Code-switched English-Hindi",
    }

    for lang, info in lang_dist.items():
        desc = names.get(lang, lang)
        lines.append(
            f"| `{lang}` | {desc} | {info['count']:,} | {info['percentage']}% |"
        )
    lines.append("")

    lines.append("## Quality & Validation Statistics\n")
    lines.append(f"- **Validations Passed**: {quality.get('validation_passed')}")
    lines.append(f"- **Validations Failed**: {quality.get('validation_failed')}")
    lines.append(f"- **Pass Rate**: {quality.get('pass_rate_percentage')}%\n")

    return "\n".join(lines)
