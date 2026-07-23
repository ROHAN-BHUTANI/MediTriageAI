"""
Execution Pipeline for the REF Benchmark & Ablation Engine.

Orchestrates global execution across all registered benchmark strategies.
Discovery -> Dispatch -> Collection -> Comparison -> Aggregation -> Summary Generation.
"""

from typing import Any
import logging
import json
from pathlib import Path
from collections import OrderedDict

from ref.benchmarks.types import (
    BenchmarkComparison,
    AblationSummary,
    BenchmarkSummary
)
from ref.benchmarks.registry import BenchmarkRegistry

logger = logging.getLogger(__name__)

class BenchmarkPipeline:
    """
    Executes the benchmarking pipeline orchestrating isolated strategies
    into a consolidated BenchmarkSummary.
    """

    def __init__(self, registry: BenchmarkRegistry, report_id: str, output_dir: str):
        self.registry = registry
        self.report_id = report_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # State tracking through pipeline
        self.context: dict[str, Any] = {}
        self.comparisons: list[BenchmarkComparison] = []
        self.ablations: list[AblationSummary] = []
        self.final_summary: BenchmarkSummary | None = None

    def execute(self, context: dict[str, Any]) -> BenchmarkSummary:
        """Execute the strict benchmarking pipeline."""
        
        logger.debug(f"Benchmark Pipeline started for Report ID: {self.report_id}")
        
        # Stages 1-5 wrapped in dispatch
        self.context = context
        self.dispatch_and_aggregate_stage()
        
        # Stage 6: Summary Generation (Benchmark Report)
        return self.summary_generation_stage()

    def dispatch_and_aggregate_stage(self) -> None:
        """
        Dispatches context to all registered strategies.
        Each strategy internally handles:
        discover -> schedule -> collect -> compare -> aggregate -> report.
        """
        logger.debug("Benchmark Pipeline: Dispatch & Aggregate Stage")
        
        suites = self.registry.get_all_suites()
        
        for suite_name in suites:
            strategies = self.registry.get_strategies_for_suite(suite_name)
            
            for strategy in strategies:
                name = strategy.get_strategy_name()
                logger.debug(f"Dispatching to benchmark strategy: {name}")
                
                report_obj = strategy.execute_lifecycle(self.context)
                
                if isinstance(report_obj, BenchmarkComparison):
                    self.comparisons.append(report_obj)
                elif isinstance(report_obj, AblationSummary):
                    self.ablations.append(report_obj)
                else:
                    logger.warning(f"Unknown return type from strategy {name}")

    def summary_generation_stage(self) -> BenchmarkSummary:
        """Generate final benchmark summary and export report."""
        logger.debug("Benchmark Pipeline: Summary Generation Stage")
        
        # Deterministic sorting inside data structures
        self.final_summary = BenchmarkSummary(
            report_id=self.report_id,
            comparisons=self.comparisons,
            ablations=self.ablations
        )
        self.final_summary.validate()
        
        # Write benchmark summary to JSON
        report_path = self.output_dir / "benchmark_summary.json"
        
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(self.final_summary.to_dict(), f, indent=4, sort_keys=True)
            
        logger.info(f"Benchmark summary written to: {report_path}")
        return self.final_summary
