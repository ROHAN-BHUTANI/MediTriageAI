"""Multilingual Translation Provider Interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class MultilingualProvider(ABC):
    """Abstract base class for multilingual clinical translation/expansion providers."""

    @abstractmethod
    def translate_text(
        self,
        text: str,
        target_lang: str,
        department: str | None = None,
        triage_level: str | None = None,
    ) -> str:
        """Translate a single clinical text into the target language.

        Args:
            text: Source English text.
            target_lang: Target language code ('hi', 'hi-Latn', 'hi-en', 'en-hi').
            department: Medical department label.
            triage_level: Severity label.

        Returns:
            Translated clinical text.
        """

    @abstractmethod
    def provider_metadata(self) -> dict[str, Any]:
        """Return provider statistics and metadata."""

    @property
    def name(self) -> str:
        return self.__class__.__name__
