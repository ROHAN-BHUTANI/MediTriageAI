"""Augmentation Plugin – Abstract Interface.

Every augmentation technique implements this interface.
Stage 6 interacts exclusively with AugmentationPlugin instances.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class AugmentedSample:
    """Container for an augmented text sample with full provenance."""

    text: str
    original_sample_id: str
    plugin_used: str
    random_seed: int
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)


class AugmentationPlugin(ABC):
    """Abstract base for all augmentation plugins."""

    @abstractmethod
    def apply(self, text: str, seed: int, **kwargs: Any) -> str:
        """Apply this augmentation to a text sample.

        Args:
            text: Input text.
            seed: Random seed for determinism.
            **kwargs: Plugin-specific options.

        Returns:
            Augmented text string.
        """

    @abstractmethod
    def plugin_metadata(self) -> dict[str, Any]:
        """Return metadata about this plugin (name, version, description)."""

    @abstractmethod
    def supported_languages(self) -> list[str]:
        """Return list of language tags this plugin can process."""

    @property
    def name(self) -> str:
        return self.__class__.__name__
