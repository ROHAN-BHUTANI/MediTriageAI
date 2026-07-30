"""
Base strategy for the REF Benchmark & Ablation Engine.

Defines the strictly enforced lifecycle for all experimental comparison strategies:
discover() -> schedule() -> collect() -> compare() -> aggregate() -> report().
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

from ref.benchmarks.types import AblationSummary, BenchmarkComparison

logger = logging.getLogger(__name__)


class BaseBenchmarkStrategy(ABC):
    """
    Abstract base class for all benchmark and ablation orchestration engines.
    Forces strict decoupling of experiment discovery, metric collection, and comparison.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

        # State tracking through lifecycle
        self.discovered_targets: list[str] = []
        self.scheduled_runs: list[str] = []
        self.collected_metrics: dict[str, Any] = {}
        self.comparisons: dict[str, Any] = {}
        self.aggregated_results: dict[str, Any] = {}
        self.final_report: BenchmarkComparison | AblationSummary | None = None

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Define identity for this strategy."""

    def execute_lifecycle(
        self, context: dict[str, Any]
    ) -> BenchmarkComparison | AblationSummary:
        """
        Immutable execution template.
        """
        logger.debug(f"Executing benchmark lifecycle: {self.__class__.__name__}")

        # Stage 1: Discover
        self.discovered_targets = self.discover(context)

        # Stage 2: Schedule (Identify required execution hashes / runs)
        self.scheduled_runs = self.schedule(self.discovered_targets)

        # Stage 3: Collect (Gather metrics for scheduled runs)
        self.collected_metrics = self.collect(self.scheduled_runs, context)

        # Stage 4: Compare (Compute pairwise or group deltas)
        self.comparisons = self.compare(self.collected_metrics)

        # Stage 5: Aggregate (Roll up into summary statistics)
        self.aggregated_results = self.aggregate(self.comparisons)

        # Stage 6: Report (Package into domain types)
        self.final_report = self.report(self.aggregated_results)

        return self.final_report

    @abstractmethod
    def discover(self, context: dict[str, Any]) -> list[str]:
        """Identify which experiments or configurations need to be benchmarked."""

    @abstractmethod
    def schedule(self, targets: list[str]) -> list[str]:
        """Determine specific execution tracking IDs."""

    @abstractmethod
    def collect(self, run_ids: list[str], context: dict[str, Any]) -> dict[str, Any]:
        """Fetch pre-computed metrics for the requested run IDs."""

    @abstractmethod
    def compare(self, metrics: dict[str, Any]) -> dict[str, Any]:
        """Compute the mathematical deltas between baseline and candidates."""

    @abstractmethod
    def aggregate(self, comparisons: dict[str, Any]) -> dict[str, Any]:
        """Compute higher order statistics across the comparison group."""

    @abstractmethod
    def report(
        self, aggregated_results: dict[str, Any]
    ) -> BenchmarkComparison | AblationSummary:
        """Wrap the outputs into strict BenchmarkComparison or AblationSummary objects."""
