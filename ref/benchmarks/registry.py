"""
Registry for the REF Benchmark & Ablation Engine.

Provides the `BenchmarkRegistry` to strictly govern experimental strategy lifecycle,
suite registration, and configurable active tracking.
"""

from typing import Any, Type
import logging
from collections import OrderedDict

from ref.benchmarks.base import BaseBenchmarkStrategy
from ref.benchmarks.types import BenchmarkValidationError

logger = logging.getLogger(__name__)

class BenchmarkRegistry:
    """
    Central orchestrator for tracking and instantiating benchmark strategies.
    Supports deterministic ordered tracking and dependency resolution.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        # Deterministic dictionaries
        self._strategies: dict[str, BaseBenchmarkStrategy] = OrderedDict()
        self._strategy_classes: dict[str, Type[BaseBenchmarkStrategy]] = {}
        self._suites: dict[str, list[str]] = {}

    def discover_strategies(self, strategy_classes: list[Type[BaseBenchmarkStrategy]]) -> None:
        """Register classes to the internal discovery dict."""
        for cls in strategy_classes:
            temp_inst = cls()
            name = temp_inst.get_strategy_name()
            self._strategy_classes[name] = cls
            logger.debug(f"Discovered benchmark strategy: {name}")

    def register(self, suite_name: str, strategy_names: list[str]) -> None:
        """
        Register specific strategies to a suite and instantiate them if enabled.
        Follows configuration-driven enable/disable logic (Stage 6).
        """
        if suite_name not in self._suites:
            self._suites[suite_name] = []
            
        engine_enabled = self.config.get("benchmark_engine_enabled", True)
        if not engine_enabled:
            logger.info("Benchmark Engine disabled globally via configuration.")
            return
            
        suite_enabled = self.config.get(f"enable_suite_{suite_name.lower()}", True)
        if not suite_enabled:
            logger.debug(f"Skipping suite {suite_name} (disabled via config).")
            return

        for s_name in strategy_names:
            strategy_enabled = self.config.get(f"enable_strategy_{s_name.lower()}", True)
            
            if not strategy_enabled:
                logger.debug(f"Skipping registration of strategy {s_name} (disabled via config).")
                continue

            if s_name not in self._strategy_classes:
                raise ValueError(f"Benchmark Strategy {s_name} not discovered.")
                
            if s_name not in self._strategies:
                cls = self._strategy_classes[s_name]
                inst = cls(self.config)
                # Dependency validation could be injected here
                self._strategies[s_name] = inst
            
            if s_name not in self._suites[suite_name]:
                self._suites[suite_name].append(s_name)
                
    def get_strategies_for_suite(self, suite_name: str) -> list[BaseBenchmarkStrategy]:
        """Fetch all initialized strategies assigned to a specific suite."""
        if suite_name not in self._suites:
            return []
        return [self._strategies[name] for name in self._suites[suite_name]]

    def get_all_suites(self) -> list[str]:
        """Return deterministic ordered list of registered suites."""
        return sorted(list(self._suites.keys()))
