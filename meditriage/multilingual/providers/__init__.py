"""Multilingual Translation Provider Factory & Registry."""

from __future__ import annotations

from typing import Any

from meditriage.multilingual.providers.base import MultilingualProvider
from meditriage.multilingual.providers.offline import OfflineMultilingualProvider

_REGISTRY: dict[str, type[MultilingualProvider]] = {
    "offline": OfflineMultilingualProvider,
}

try:
    from meditriage.multilingual.providers.gemini import GeminiMultilingualProvider

    _REGISTRY["gemini"] = GeminiMultilingualProvider
except ImportError:
    pass

try:
    from meditriage.multilingual.providers.openai import OpenAIMultilingualProvider

    _REGISTRY["openai"] = OpenAIMultilingualProvider
except ImportError:
    pass


def register_provider(name: str, cls: type[MultilingualProvider]) -> None:
    _REGISTRY[name.lower()] = cls


def get_provider(name: str, **kwargs: Any) -> MultilingualProvider:
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown provider '{name}'. Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[key](**kwargs)


def list_providers() -> list[str]:
    return list(_REGISTRY.keys())
