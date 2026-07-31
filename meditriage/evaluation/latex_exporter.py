"""Production LaTeX Table Exporter for Research Publications."""

from __future__ import annotations

from typing import Any


class LatexTableExporter:
    """Exports benchmark summaries, ablation studies, and robustness metrics into publication-ready LaTeX tables."""

    @staticmethod
    def export_ablation_table(results: list[dict[str, Any]]) -> str:
        """Export ablation study results into a LaTeX booktabs table string."""
        latex = [
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{Ablation Study of Dataset Pipeline Components on MediTriageAI Benchmark}",
            "\\label{tab:ablation}",
            "\\begin{booktabs}{lcccc}",
            "\\toprule",
            "Experiment Variant & Accuracy & Macro F1 & Weighted F1 & Balanced Acc. \\\\",
            "\\midrule",
        ]

        for row in results:
            name = str(row.get("experiment_name", "")).replace("_", "\\_")
            acc = row.get("accuracy", 0.0)
            f1 = row.get("macro_f1", 0.0)
            wf1 = row.get("weighted_f1", 0.0)
            bacc = row.get("balanced_accuracy", 0.0)
            latex.append(f"{name} & {acc:.4f} & {f1:.4f} & {wf1:.4f} & {bacc:.4f} \\\\")

        latex.extend(
            [
                "\\bottomrule",
                "\\end{booktabs}",
                "\\end{table}",
            ]
        )
        return "\n".join(latex)

    @staticmethod
    def export_backbone_table(results: list[dict[str, Any]]) -> str:
        """Export backbone comparison results into a LaTeX table string."""
        latex = [
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{Multilingual Clinical Transformer Backbone Comparison}",
            "\\label{tab:backbones}",
            "\\begin{booktabs}{lcccc}",
            "\\toprule",
            "Backbone Model & Accuracy & Macro F1 & Top-2 Acc. & ECE \\\\",
            "\\midrule",
        ]

        for row in results:
            name = str(row.get("model", "")).replace("_", "\\_")
            acc = row.get("accuracy", 0.0)
            f1 = row.get("macro_f1", 0.0)
            top2 = row.get("top2_accuracy", 0.0)
            ece = row.get("calibration_error", 0.0)
            latex.append(f"{name} & {acc:.4f} & {f1:.4f} & {top2:.4f} & {ece:.4f} \\\\")

        latex.extend(
            [
                "\\bottomrule",
                "\\end{booktabs}",
                "\\end{table}",
            ]
        )
        return "\n".join(latex)
