import pytest
import os
import shutil
import json
from pathlib import Path

from ref.benchmarks.types import (
    BenchmarkDefinition,
    BenchmarkRun,
    BenchmarkComparison,
    BenchmarkSuite,
    AblationDefinition,
    AblationRun,
    AblationSummary,
    BenchmarkSummary,
    BenchmarkValidationError
)
from ref.benchmarks.base import BaseBenchmarkStrategy
from ref.benchmarks.registry import BenchmarkRegistry
from ref.benchmarks.pipeline import BenchmarkPipeline
from ref.benchmarks.strategies import BaselineBenchmark, AblationBenchmark

@pytest.fixture
def temp_output(tmp_path):
    output = tmp_path / "bench_out"
    output.mkdir()
    yield output
    if output.exists():
        shutil.rmtree(output)

class MockStrategy(BaseBenchmarkStrategy):
    def get_strategy_name(self) -> str:
        return "MockStrategy"
        
    def discover(self, context: dict) -> list[str]:
        return ["exp_A"]
        
    def schedule(self, targets: list[str]) -> list[str]:
        return targets
        
    def collect(self, run_ids: list[str], context: dict) -> dict:
        return {rid: {"f1": 0.90} for rid in run_ids}
        
    def compare(self, metrics: dict) -> dict:
        return {"delta_f1": 0.05}
        
    def aggregate(self, comparisons: dict) -> dict:
        return comparisons
        
    def report(self, aggregated_results: dict) -> BenchmarkComparison:
        br_base = BenchmarkRun(experiment_id="base", metrics={"f1": 0.85})
        br_cand = BenchmarkRun(experiment_id="exp_A", metrics={"f1": 0.90})
        comp = BenchmarkComparison(
            comparison_id="mock_comp",
            base_run=br_base,
            candidate_runs=[br_cand],
            deltas=aggregated_results
        )
        return comp

def test_benchmark_data_structures():
    bd = BenchmarkDefinition(name="task1", target_metric="f1", baseline_experiment_id="base")
    bd.validate()
    
    with pytest.raises(BenchmarkValidationError):
        BenchmarkDefinition(name="", target_metric="f1", baseline_experiment_id="").validate()

    br = BenchmarkRun(experiment_id="e1", metrics={"f1": 0.9})
    br.validate()
    
    comp = BenchmarkComparison(comparison_id="c1", base_run=br, candidate_runs=[br], deltas={"d": 0})
    comp.validate()
    
    ab_def = AblationDefinition(ablation_name="ab1", removed_components=["x"])
    ab_run = AblationRun(experiment_id="e2", ablation=ab_def, metrics={})
    
    ab_sum = AblationSummary(study_name="s1", baseline_run=br, ablation_runs=[ab_run], impact_scores={"i": -0.1})
    ab_sum.validate()
    
    summary = BenchmarkSummary(report_id="r1", comparisons=[comp], ablations=[ab_sum])
    summary.validate()
    
    # Deterministic Hash Check
    assert hasattr(summary, "summary_hash")
    d = summary.to_dict()
    assert "summary_hash" in d
    
    summary2 = BenchmarkSummary.from_dict(d)
    assert summary2.summary_hash == summary.summary_hash

def test_strategy_interface_compliance():
    strategy = MockStrategy()
    result = strategy.execute_lifecycle({})
    
    assert isinstance(result, BenchmarkComparison)
    assert result.comparison_id == "mock_comp"
    assert result.deltas["delta_f1"] == 0.05

def test_registry_integrity_and_config():
    config = {
        "benchmark_engine_enabled": True,
        "enable_suite_evals": True,
        "enable_strategy_baselinebenchmark": False,
        "enable_strategy_ablationbenchmark": True
    }
    
    registry = BenchmarkRegistry(config)
    registry.discover_strategies([BaselineBenchmark, AblationBenchmark])
    
    registry.register("evals", ["BaselineBenchmark", "AblationBenchmark"])
    
    strategies = registry.get_strategies_for_suite("evals")
    assert len(strategies) == 1
    assert isinstance(strategies[0], AblationBenchmark)

def test_benchmark_pipeline_and_manifest(temp_output):
    registry = BenchmarkRegistry({})
    registry.discover_strategies([MockStrategy, AblationBenchmark])
    registry.register("test_suite", ["MockStrategy", "AblationBenchmark"])
    
    pipeline = BenchmarkPipeline(registry, report_id="REP_TEST", output_dir=str(temp_output))
    summary = pipeline.execute({})
    
    assert isinstance(summary, BenchmarkSummary)
    assert len(summary.comparisons) == 1
    assert len(summary.ablations) == 1
    
    # Verify benchmark_summary.json generation
    bench_file = temp_output / "benchmark_summary.json"
    assert bench_file.exists()
    
    with open(bench_file, "r") as f:
        data = json.load(f)
        assert data["report_id"] == "REP_TEST"
        assert "summary_hash" in data
        assert len(data["comparisons"]) == 1
        assert len(data["ablations"]) == 1
