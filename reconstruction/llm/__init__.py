"""LLM Provider – Abstract Interface and Provider Registry.

Every LLM backend (Gemini, OpenAI, local) implements LLMProvider.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
import hashlib


@dataclass
class GeneratedSample:
    """Container for an LLM-generated sample with full provenance."""

    text: str
    department: str
    source_sample_id: str
    generation_prompt_hash: str
    provider: str
    generation_id: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)


class LLMProvider(ABC):
    """Abstract base for LLM generation providers."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        n: int = 1,
        **kwargs: Any,
    ) -> list[str]:
        """Generate n text completions from the prompt.

        Args:
            prompt: Full generation prompt.
            n: Number of completions.

        Returns:
            List of generated text strings.
        """

    @abstractmethod
    def validate(self, text: str, department: str) -> bool:
        """Validate that a generated text is clinically plausible.

        Args:
            text: Generated text.
            department: Expected department label.

        Returns:
            True if the sample passes validation.
        """

    @abstractmethod
    def provider_metadata(self) -> dict[str, Any]:
        """Return provider metadata (name, model, version)."""

    @property
    def name(self) -> str:
        return self.__class__.__name__


def hash_prompt(prompt: str) -> str:
    """Compute a deterministic hash of a generation prompt."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


# ── Provider Registry ────────────────────────────────────────────────────

_REGISTRY: dict[str, type[LLMProvider]] = {}


def register_provider(name: str, cls: type[LLMProvider]) -> None:
    _REGISTRY[name] = cls


def get_provider(name: str, **kwargs: Any) -> LLMProvider:
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown LLM provider '{name}'. Registered: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[name](**kwargs)


def list_providers() -> list[str]:
    return list(_REGISTRY.keys())
