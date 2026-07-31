"""Diversity Report Generator.

Produces a comprehensive final report after reconstruction in
Markdown, CSV, and JSON formats.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from reconstruction.config import ReconstructionConfig

logger = logging.getLogger(__name__)


def generate_diversity_report(df: pd.DataFrame, cfg: ReconstructionConfig) -> dict:
    """Generate the final diversity report across all axes.

    Args:
        df: Final validated DataFrame.
        cfg: Reconstruction configuration.

    Returns:
        Report dictionary (also written to disk).
    """
    out_dir = Path(cfg.output_directory)
    out_dir.mkdir(parents=True, exist_ok=True)

    report: dict = {}

    # ── Class counts ──────────────────────────────────────────────────────
    dept_counts = df["department"].value_counts().to_dict() if "department" in df.columns else {}
    report["class_counts"] = dept_counts

    # ── Language counts ───────────────────────────────────────────────────
    lang_counts = df["language"].value_counts().to_dict() if "language" in df.columns else {}
    report["language_counts"] = lang_counts

    # ── Phenotype counts ──────────────────────────────────────────────────
    if "cluster_id" in df.columns:
        pheno = {}
        for dept, grp in df.groupby("department"):
            pheno[dept] = int(grp["cluster_id"].nunique())
        report["phenotype_counts"] = pheno
    else:
        report["phenotype_counts"] = {}

    # ── Augmentation statistics ───────────────────────────────────────────
    if "dataset_source" in df.columns:
        source_counts = df["dataset_source"].value_counts().to_dict()
        aug_sources = {k: v for k, v in source_counts.items() if k.startswith("augmented_")}
        syn_sources = {k: v for k, v in source_counts.items() if k.startswith("synthetic_")}
        orig_count = sum(v for k, v in source_counts.items()
                         if not k.startswith(("augmented_", "synthetic_")))
        report["augmentation_statistics"] = {
            "plugin_usage": aug_sources,
            "total_augmented": sum(aug_sources.values()),
        }
        report["synthetic_generation_statistics"] = {
            "provider_usage": syn_sources,
            "total_synthetic": sum(syn_sources.values()),
        }
        report["composition"] = {
            "original": orig_count,
            "augmented": sum(aug_sources.values()),
            "synthetic": sum(syn_sources.values()),
            "total": len(df),
        }
    else:
        report["augmentation_statistics"] = {}
        report["synthetic_generation_statistics"] = {}

    # ── Diversity scores ──────────────────────────────────────────────────
    div_cols = [c for c in df.columns if c.startswith("div_") or c == "diversity_score"]
    if div_cols:
        div_stats = {}
        for col in div_cols:
            if col in df.columns:
                div_stats[col] = {
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                }
        report["diversity_scores"] = div_stats

    # ── Provenance columns ────────────────────────────────────────────────
    prov_cols = [c for c in df.columns if c.startswith("_provenance_")]
    report["provenance_columns"] = prov_cols

    # ── Summary ───────────────────────────────────────────────────────────
    report["final_summary"] = {
        "total_rows": len(df),
        "total_departments": df["department"].nunique() if "department" in df.columns else 0,
        "total_languages": df["language"].nunique() if "language" in df.columns else 0,
        "target_class_size": cfg.target_class_size,
        "balanced": all(v == cfg.target_class_size for v in dept_counts.values()) if dept_counts else False,
    }

    # ── Write JSON ────────────────────────────────────────────────────────
    json_path = out_dir / "diversity_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # ── Write CSV ─────────────────────────────────────────────────────────
    csv_path = out_dir / "diversity_report_classes.csv"
    class_df = pd.DataFrame([
        {"department": k, "count": v, "target": cfg.target_class_size,
         "balanced": v == cfg.target_class_size}
        for k, v in dept_counts.items()
    ])
    if not class_df.empty:
        class_df.to_csv(csv_path, index=False)

    # ── Write Markdown ────────────────────────────────────────────────────
    md_path = out_dir / "diversity_report.md"
    md = _build_markdown(report, cfg)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)

    logger.info("Diversity report written to %s", out_dir)
    return report


def _build_markdown(report: dict, cfg: ReconstructionConfig) -> str:
    """Build a Markdown string from the report dict."""
    lines = ["# Dataset Reconstruction – Diversity Report\n"]

    # Summary
    s = report.get("final_summary", {})
    lines.append("## Summary\n")
    lines.append(f"- **Total rows**: {s.get('total_rows', 'N/A')}")
    lines.append(f"- **Departments**: {s.get('total_departments', 'N/A')}")
    lines.append(f"- **Languages**: {s.get('total_languages', 'N/A')}")
    lines.append(f"- **Target class size**: {s.get('target_class_size', 'N/A')}")
    lines.append(f"- **Balanced**: {'✅' if s.get('balanced') else '❌'}\n")

    # Class counts
    lines.append("## Class Counts\n")
    lines.append("| Department | Count | Target | Balanced |")
    lines.append("|-----------|-------|--------|----------|")
    for dept, count in report.get("class_counts", {}).items():
        bal = "✅" if count == cfg.target_class_size else "❌"
        lines.append(f"| {dept} | {count} | {cfg.target_class_size} | {bal} |")
    lines.append("")

    # Language counts
    lines.append("## Language Distribution\n")
    for lang, count in report.get("language_counts", {}).items():
        lines.append(f"- **{lang}**: {count}")
    lines.append("")

    # Composition
    comp = report.get("composition", {})
    if comp:
        lines.append("## Data Composition\n")
        lines.append(f"- Original: {comp.get('original', 0)}")
        lines.append(f"- Augmented: {comp.get('augmented', 0)}")
        lines.append(f"- Synthetic: {comp.get('synthetic', 0)}")
        lines.append(f"- **Total**: {comp.get('total', 0)}\n")

    # Diversity scores
    div = report.get("diversity_scores", {})
    if div:
        lines.append("## Diversity Scores\n")
        lines.append("| Metric | Mean | Std | Min | Max |")
        lines.append("|--------|------|-----|-----|-----|")
        for metric, stats in div.items():
            lines.append(
                f"| {metric} | {stats['mean']:.4f} | {stats['std']:.4f} "
                f"| {stats['min']:.4f} | {stats['max']:.4f} |"
            )
        lines.append("")

    return "\n".join(lines)
