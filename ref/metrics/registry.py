"""
Registry for the REF Metrics Engine.

Provides the `MetricRegistry` to strictly govern provider lifecycle,
dependency validation, grouping, and configurable active tracking.
"""

import logging
from collections import OrderedDict
from typing import Any

from ref.metrics.base import BaseMetricProvider
from ref.metrics.types import MetricValidationError

logger = logging.getLogger(__name__)


class MetricRegistry:
    """
    Central orchestrator for tracking and instantiating metric providers.
    Supports deterministic ordered tracking and dependency resolution.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        # Use OrderedDict to enforce deterministic execution ordering
        self._providers: dict[str, BaseMetricProvider] = OrderedDict()
        self._provider_classes: dict[str, type[BaseMetricProvider]] = {}
        self._groups: dict[str, list[str]] = {}

    def discover_providers(
        self, provider_classes: list[type[BaseMetricProvider]]
    ) -> None:
        """Register classes to the internal discovery dict."""
        for cls in provider_classes:
            # We instantiate purely to extract the metadata
            temp_inst = cls()
            meta = temp_inst.get_metadata()
            self._provider_classes[meta.name] = cls
            logger.debug(f"Discovered metric provider: {meta.name} (v{meta.version})")

    def register(self, group_name: str, provider_names: list[str]) -> None:
        """
        Register specific providers to a group and instantiate them if they are enabled.
        Follows configuration-driven enable/disable logic (Stage 6).
        """
        if group_name not in self._groups:
            self._groups[group_name] = []

        # Global metric enable/disable logic based on config
        engine_enabled = self.config.get("metrics_engine_enabled", True)
        if not engine_enabled:
            logger.info("Metrics Engine disabled globally via configuration.")
            return

        for p_name in provider_names:
            # Stage 6 Configuration: skip if provider disabled
            provider_enabled = self.config.get(
                f"enable_provider_{p_name.lower()}", True
            )
            group_enabled = self.config.get(f"enable_group_{group_name.lower()}", True)

            if not provider_enabled or not group_enabled:
                logger.debug(
                    f"Skipping registration of provider {p_name} (disabled via config)."
                )
                continue

            if p_name not in self._provider_classes:
                raise ValueError(f"Provider {p_name} not discovered.")

            if p_name not in self._providers:
                cls = self._provider_classes[p_name]
                inst = cls(self.config)

                # Check dependencies
                meta = inst.get_metadata()
                for dep in meta.dependencies:
                    if dep not in self._providers:
                        raise MetricValidationError(
                            f"Provider '{p_name}' depends on '{dep}' which is not registered/enabled before it."
                        )

                self._providers[p_name] = inst

            if p_name not in self._groups[group_name]:
                self._groups[group_name].append(p_name)

    def get_providers_for_group(self, group_name: str) -> list[BaseMetricProvider]:
        """Fetch all initialized providers assigned to a specific group."""
        if group_name not in self._groups:
            return []
        return [self._providers[name] for name in self._groups[group_name]]

    def get_all_groups(self) -> list[str]:
        """Return deterministic ordered list of registered groups."""
        return sorted(list(self._groups.keys()))
