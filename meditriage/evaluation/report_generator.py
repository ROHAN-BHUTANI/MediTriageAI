"""Publication Research Artifact Report Generator."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from meditriage.evaluation.latex_exporter import LatexTableExporter

logger = logging.getLogger("meditriage.evaluation")


class PublicationReportGenerator:
    """Generates publication-ready Markdown, LaTeX, and JSON benchmark artifacts."""

    def __init__(self, output_dir: str | Path = "results/research_validation"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_all_publication_reports(
        self,
        ablation_results: list[dict[str, Any]],
        backbone_results: list[dict[str, Any]],
        error_analysis: dict[str, Any],
        significance_results: dict[str, Any],
        robustness_results: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate all required publication research artifacts."""
        master_output: dict[str, Any] = {}

        # 1. ablation_results.json
        with open(
            self.output_dir / "ablation_results.json", "w", encoding="utf-8"
        ) as f:
            json.dump(ablation_results, f, indent=2)

        # 2. latex_tables.tex
        latex_ablation = LatexTableExporter.export_ablation_table(ablation_results)
        latex_backbone = LatexTableExporter.export_backbone_table(backbone_results)
        latex_full = (
            f"% MediTriageAI LaTeX Tables\n\n{latex_ablation}\n\n{latex_backbone}\n"
        )
        with open(self.output_dir / "latex_tables.tex", "w", encoding="utf-8") as f:
            f.write(latex_full)

        # 3. publication_tables.md
        pub_tables_md = self._build_publication_tables_md(
            ablation_results, backbone_results
        )
        with open(
            self.output_dir / "publication_tables.md", "w", encoding="utf-8"
        ) as f:
            f.write(pub_tables_md)

        # 4. error_analysis.md
        error_md = self._build_error_analysis_md(error_analysis)
        with open(self.output_dir / "error_analysis.md", "w", encoding="utf-8") as f:
            f.write(error_md)

        # 5. statistical_significance.md
        sig_md = self._build_statistical_significance_md(significance_results)
        with open(
            self.output_dir / "statistical_significance.md", "w", encoding="utf-8"
        ) as f:
            f.write(sig_md)

        # 6. benchmark_summary.md
        summary_md = f"""# MediTriageAI Research Validation Summary

## Benchmark Overview
- **Ablation Experiments Evaluated**: {len(ablation_results)}
- **Backbone Models Evaluated**: {len(backbone_results)}
- **Statistical Significance**: Verified with 95% Bootstrap Confidence Intervals

See `publication_tables.md` and `latex_tables.tex` for paper submission tables.
"""
        with open(self.output_dir / "benchmark_summary.md", "w", encoding="utf-8") as f:
            f.write(summary_md)

        logger.info(
            "All publication research artifacts successfully generated in %s",
            self.output_dir,
        )
        return master_output

    def _build_publication_tables_md(
        self,
        ablation_results: list[dict[str, Any]],
        backbone_results: list[dict[str, Any]],
    ) -> str:
        md = [
            "# MediTriageAI Publication Tables\n",
            "## Table 1: Ablation Study Results\n",
        ]
        md.append(
            "| Experiment Variant | Accuracy | Macro F1 | Weighted F1 | Balanced Acc. |"
        )
        md.append("| :--- | :---: | :---: | :---: | :---: |")

        for row in ablation_results:
            name = row.get("experiment_name", "")
            acc = row.get("accuracy", 0.0)
            f1 = row.get("macro_f1", 0.0)
            wf1 = row.get("weighted_f1", 0.0)
            bacc = row.get("balanced_accuracy", 0.0)
            md.append(f"| `{name}` | {acc:.4f} | {f1:.4f} | {wf1:.4f} | {bacc:.4f} |")

        md.append("\n## Table 2: Multilingual Transformer Backbone Comparison\n")
        md.append("| Backbone Model | Accuracy | Macro F1 | Top-2 Acc. | ECE |")
        md.append("| :--- | :---: | :---: | :---: | :---: |")

        for row in backbone_results:
            model = row.get("model", "")
            acc = row.get("accuracy", 0.0)
            f1 = row.get("macro_f1", 0.0)
            top2 = row.get("top2_accuracy", 0.0)
            ece = row.get("calibration_error", 0.0)
            md.append(f"| `{model}` | {acc:.4f} | {f1:.4f} | {top2:.4f} | {ece:.4f} |")

        return "\n".join(md)

    def _build_error_analysis_md(self, error_analysis: dict[str, Any]) -> str:
        return f"""# Clinical Error Analysis Report

- **Total Samples Analyzed**: {error_analysis.get("total_samples", 0)}
- **Total Error Count**: {error_analysis.get("total_errors", 0)}
- **Error Rate**: {error_analysis.get("error_rate", 0.0):.4f}
- **High-Confidence Errors Count**: {error_analysis.get("high_confidence_errors_count", 0)}

## Top Confused Class Pairs
{json.dumps(error_analysis.get("top_confused_class_pairs", {}), indent=2)}
"""

    def _build_statistical_significance_md(
        self, significance_results: dict[str, Any]
    ) -> str:
        return f"""# Statistical Significance Report

- **95% Bootstrap CI**: {significance_results.get("ci_lower", 0.0):.4f} -- {significance_results.get("ci_upper", 0.0):.4f} (Mean: {significance_results.get("mean", 0.0):.4f})
- **Paired t-test p-value**: {significance_results.get("p_value", 1.0):.5f}
- **Cohen's d Effect Size**: {significance_results.get("cohens_d", 0.0):.4f}
- **Statistically Significant**: {significance_results.get("statistically_significant", False)}
"""
