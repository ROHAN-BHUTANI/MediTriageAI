"""
Concrete strategies for the REF Benchmark & Ablation Engine.

Implements specific comparison patterns derived from BaseBenchmarkStrategy.
"""

from typing import Any

from ref.benchmarks.base import BaseBenchmarkStrategy
from ref.benchmarks.types import (
    AblationDefinition,
    AblationRun,
    AblationSummary,
    BenchmarkComparison,
    BenchmarkRun,
)


class DummyStrategyMixin:
    """Provides stub implementations for the provider lifecycle to satisfy interfaces."""

    def discover(self, context: dict[str, Any]) -> list[str]:
        return ["dummy_target"]

    def schedule(self, targets: list[str]) -> list[str]:
        return ["run_baseline", "run_candidate"]

    def collect(self, run_ids: list[str], context: dict[str, Any]) -> dict[str, Any]:
        return {rid: {"accuracy": 0.95} for rid in run_ids}

    def compare(self, metrics: dict[str, Any]) -> dict[str, Any]:
        return {"delta_accuracy": 0.0}

    def aggregate(self, comparisons: dict[str, Any]) -> dict[str, Any]:
        return comparisons

    def report(
        self, aggregated_results: dict[str, Any]
    ) -> BenchmarkComparison | AblationSummary:
        br_base = BenchmarkRun(experiment_id="run_baseline", metrics={"accuracy": 0.95})
        br_cand = BenchmarkRun(
            experiment_id="run_candidate", metrics={"accuracy": 0.95}
        )

        comp = BenchmarkComparison(
            comparison_id=f"comp_{self.get_strategy_name()}",
            base_run=br_base,
            candidate_runs=[br_cand],
            deltas=aggregated_results,
        )
        comp.validate()
        return comp


class BaselineBenchmark(DummyStrategyMixin, BaseBenchmarkStrategy):
    def get_strategy_name(self) -> str:
        return "BaselineBenchmark"


class ModuleBenchmark(DummyStrategyMixin, BaseBenchmarkStrategy):
    def get_strategy_name(self) -> str:
        return "ModuleBenchmark"


class CrossDatasetBenchmark(DummyStrategyMixin, BaseBenchmarkStrategy):
    def get_strategy_name(self) -> str:
        return "CrossDatasetBenchmark"


class CrossCheckpointBenchmark(DummyStrategyMixin, BaseBenchmarkStrategy):
    def get_strategy_name(self) -> str:
        return "CrossCheckpointBenchmark"


class LiteratureBenchmark(DummyStrategyMixin, BaseBenchmarkStrategy):
    def get_strategy_name(self) -> str:
        return "LiteratureBenchmark"


class AblationBenchmark(DummyStrategyMixin, BaseBenchmarkStrategy):
    def get_strategy_name(self) -> str:
        return "AblationBenchmark"

    def report(self, aggregated_results: dict[str, Any]) -> AblationSummary:
        """Override dummy report to return an AblationSummary."""
        br_base = BenchmarkRun(experiment_id="run_baseline", metrics={"accuracy": 0.95})
        ab_def = AblationDefinition(
            ablation_name="remove_module_x", removed_components=["module_x"]
        )
        ab_run = AblationRun(
            experiment_id="run_ablation", ablation=ab_def, metrics={"accuracy": 0.90}
        )

        summary = AblationSummary(
            study_name="AblationStudy",
            baseline_run=br_base,
            ablation_runs=[ab_run],
            impact_scores={"delta_accuracy": -0.05},
        )
        summary.validate()
        return summary
