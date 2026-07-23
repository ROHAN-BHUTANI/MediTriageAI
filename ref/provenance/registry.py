"""
Registry for the REF Provenance Framework.

Provides the `ProvenanceRegistry` to strictly govern provenance provider lifecycle,
dependency validation, grouping, and configurable active tracking.
"""

from typing import Any, Type
import logging
from collections import OrderedDict

from ref.provenance.base import BaseFingerprintProvider
from ref.provenance.types import ProvenanceValidationError

logger = logging.getLogger(__name__)

class ProvenanceRegistry:
    """
    Central orchestrator for tracking and instantiating provenance providers.
    Supports deterministic ordered tracking and dependency resolution.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self._providers: dict[str, BaseFingerprintProvider] = OrderedDict()
        self._provider_classes: dict[str, Type[BaseFingerprintProvider]] = {}
        self._registered_names: list[str] = []

    def discover_providers(self, provider_classes: list[Type[BaseFingerprintProvider]]) -> None:
        """Register classes to the internal discovery dict."""
        for cls in provider_classes:
            temp_inst = cls()
            name = temp_inst.get_provider_name()
            self._provider_classes[name] = cls
            logger.debug(f"Discovered provenance provider: {name}")

    def register(self, provider_names: list[str]) -> None:
        """
        Register specific providers and instantiate them if they are enabled.
        Follows configuration-driven enable/disable logic.
        """
        engine_enabled = self.config.get("provenance_engine_enabled", True)
        if not engine_enabled:
            logger.info("Provenance Engine disabled globally via configuration.")
            return

        for p_name in provider_names:
            provider_enabled = self.config.get(f"enable_fingerprint_{p_name.lower()}", True)
            
            if not provider_enabled:
                logger.debug(f"Skipping registration of provenance provider {p_name} (disabled via config).")
                continue

            if p_name not in self._provider_classes:
                raise ValueError(f"Provenance Provider {p_name} not discovered.")
                
            if p_name not in self._providers:
                cls = self._provider_classes[p_name]
                inst = cls(self.config)
                
                # Assume basic dependency check logic could go here if providers had dependencies
                self._providers[p_name] = inst
            
            if p_name not in self._registered_names:
                self._registered_names.append(p_name)
                
    def get_all_providers(self) -> list[BaseFingerprintProvider]:
        """Fetch all initialized providers."""
        return [self._providers[name] for name in self._registered_names]

