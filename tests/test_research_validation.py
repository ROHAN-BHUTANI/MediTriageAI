"""Unit tests for Research Validation, Evaluation, and Publication Artifact Generation.

Covers:
  - StatisticalSignificanceEngine (bootstrap 95% CIs, paired t-test, Cohen's d effect size)
  - RobustnessEvaluator (synthetic typos, code-switching, Hinglish, noise degradation)
  - ClinicalErrorAnalyzer (error categorization, high-confidence errors, class confusion pairs)
  - LatexTableExporter (LaTeX booktabs table formatting)
  - PublicationReportGenerator (publication_tables.md, latex_tables.tex, benchmark_summary.md, etc.)
  - ResearchEvaluator & ResearchBenchmarkSuite
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from meditriage.evaluation.benchmark_suite import ResearchBenchmarkSuite
from meditriage.evaluation.error_analysis import ClinicalErrorAnalyzer
from meditriage.evaluation.evaluator import ResearchEvaluator
from meditriage.evaluation.latex_exporter import LatexTableExporter
from meditriage.evaluation.report_generator import PublicationReportGenerator
from meditriage.evaluation.robustness import RobustnessEvaluator
from meditriage.evaluation.significance import StatisticalSignificanceEngine

# ─── Significance Engine Tests ─────────────────────────────────────────────


class TestStatisticalSignificanceEngine:
    def test_bootstrap_confidence_interval(self):
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        y_pred = np.array([0, 1, 2, 0, 1, 1, 0, 1, 2, 0])

        mean, low, high = StatisticalSignificanceEngine.bootstrap_confidence_interval(
            y_true, y_pred, metric="macro_f1", num_bootstraps=100
        )
        assert 0.0 <= low <= mean <= high <= 1.0

    def test_paired_significance_test(self):
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        baseline = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        model = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])

        res = StatisticalSignificanceEngine.paired_significance_test(
            y_true, baseline, model
        )
        assert "p_value" in res
        assert "cohens_d" in res
        assert res["cohens_d"] > 0


# ─── Robustness Evaluator Tests ────────────────────────────────────────────


class TestRobustnessEvaluator:
    def test_inject_synthetic_typos(self):
        text = "Patient has severe chest pain."
        typo_text = RobustnessEvaluator.inject_synthetic_typos(text, typo_rate=0.2)
        assert isinstance(typo_text, str)
        assert len(typo_text) == len(text)

    def test_evaluate_linguistic_robustness(self):
        texts = ["Text 1", "Text 2", "Text 3"]
        langs = ["en", "hi-Latn", "hi-en"]
        y_true = np.array([0, 1, 2])
        y_pred = np.array([0, 1, 2])

        res = RobustnessEvaluator.evaluate_linguistic_robustness(
            texts, langs, y_true, y_pred
        )
        assert "en" in res
        assert "hi-Latn" in res
        assert res["en"]["accuracy"] == 1.0

    def test_evaluate_noise_robustness(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        clean = np.array([0, 1, 2, 0, 1, 2])
        noisy = np.array([0, 1, 1, 0, 1, 2])

        res = RobustnessEvaluator.evaluate_noise_robustness(y_true, clean, noisy)
        assert "performance_drop" in res
        assert "retention_rate" in res


# ─── Error Analyzer Tests ──────────────────────────────────────────────────


class TestClinicalErrorAnalyzer:
    def test_analyze_errors(self):
        texts = ["Chest pain", "Fever", "Headache"]
        y_true = np.array([0, 1, 2])
        y_pred = np.array([0, 2, 2])
        probs = np.array(
            [
                [0.9, 0.05, 0.05],
                [0.1, 0.05, 0.85],  # High confidence error
                [0.05, 0.05, 0.9],
            ]
        )
        class_names = ["CARDIO", "PEDS", "NEURO"]

        res = ClinicalErrorAnalyzer.analyze_errors(
            texts, y_true, y_pred, probs, class_names
        )
        assert res["total_errors"] == 1
        assert res["high_confidence_errors_count"] == 1
        assert "PEDS -> NEURO" in res["top_confused_class_pairs"]


# ─── LaTeX & Report Generator Tests ────────────────────────────────────────


class TestLatexAndReportGenerator:
    def test_latex_table_exporter(self):
        ablation_data = [
            {
                "experiment_name": "baseline",
                "accuracy": 0.80,
                "macro_f1": 0.78,
                "weighted_f1": 0.79,
                "balanced_accuracy": 0.79,
            }
        ]
        latex_str = LatexTableExporter.export_ablation_table(ablation_data)
        assert "\\begin{table}" in latex_str
        assert "\\toprule" in latex_str
        assert "baseline" in latex_str

    def test_publication_report_generator(self, tmp_path: Path):
        gen = PublicationReportGenerator(output_dir=tmp_path / "reports")
        ablation_data = [
            {
                "experiment_name": "baseline",
                "accuracy": 0.80,
                "macro_f1": 0.78,
                "weighted_f1": 0.79,
                "balanced_accuracy": 0.79,
            }
        ]
        backbone_data = [
            {
                "model": "xlm-roberta-base",
                "accuracy": 0.91,
                "macro_f1": 0.90,
                "top2_accuracy": 0.96,
                "calibration_error": 0.03,
            }
        ]
        err_data = {
            "total_samples": 10,
            "total_errors": 1,
            "error_rate": 0.1,
            "high_confidence_errors_count": 0,
        }
        sig_data = {
            "mean": 0.90,
            "ci_lower": 0.88,
            "ci_upper": 0.92,
            "p_value": 0.001,
            "cohens_d": 0.7,
            "statistically_significant": True,
        }

        gen.generate_all_publication_reports(
            ablation_data, backbone_data, err_data, sig_data
        )

        out_dir = tmp_path / "reports"
        assert (out_dir / "publication_tables.md").exists()
        assert (out_dir / "latex_tables.tex").exists()
        assert (out_dir / "benchmark_summary.md").exists()
        assert (out_dir / "ablation_results.json").exists()
        assert (out_dir / "error_analysis.md").exists()
        assert (out_dir / "statistical_significance.md").exists()


# ─── Evaluator & Suite Tests ──────────────────────────────────────────────


class TestEvaluatorAndSuite:
    def test_research_evaluator(self):
        texts = ["Sample 1", "Sample 2"]
        y_true = np.array([0, 1])
        y_pred = np.array([0, 1])
        probs = np.array([[0.9, 0.1], [0.2, 0.8]])

        res = ResearchEvaluator.evaluate_model(texts, y_true, y_pred, probs)
        assert "metrics" in res
        assert "error_analysis" in res
        assert "significance" in res

    def test_research_benchmark_suite(self, tmp_path: Path):
        suite = ResearchBenchmarkSuite(output_dir=tmp_path / "suite_out")
        res = suite.run_full_benchmark_suite()

        assert "ablation" in res
        assert "backbone" in res
        assert (tmp_path / "suite_out" / "publication_tables.md").exists()
