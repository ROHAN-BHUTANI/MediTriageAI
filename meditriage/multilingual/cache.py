"""Persistent Thread-Safe Cache for Multilingual Dataset Expansion."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class MultilingualCache:
    """Persistent thread-safe cache mapping text hashes to translated outputs."""

    def __init__(self, cache_dir: Path | str = "results/multilingual_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "translation_cache.json"
        self._lock = threading.Lock()
        self._store: dict[str, dict[str, Any]] = {}
        self.load()

    def _compute_key(self, text: str, target_lang: str, provider_name: str) -> str:
        payload = f"{provider_name}:{target_lang}:{text.strip()}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(
        self, text: str, target_lang: str, provider_name: str
    ) -> dict[str, Any] | None:
        key = self._compute_key(text, target_lang, provider_name)
        with self._lock:
            return self._store.get(key)

    def set(
        self,
        text: str,
        target_lang: str,
        provider_name: str,
        translated_text: str,
        validated: bool = True,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        key = self._compute_key(text, target_lang, provider_name)
        val = {
            "source_text": text,
            "target_lang": target_lang,
            "provider": provider_name,
            "translated_text": translated_text,
            "validated": validated,
            "metrics": metrics or {},
        }
        with self._lock:
            self._store[key] = val

    def load(self) -> None:
        with self._lock:
            if self.cache_file.exists():
                try:
                    with open(self.cache_file, "r", encoding="utf-8") as f:
                        self._store = json.load(f)
                    logger.info(
                        "Loaded %d items from cache %s",
                        len(self._store),
                        self.cache_file,
                    )
                except Exception as exc:
                    logger.warning("Failed to load cache %s: %s", self.cache_file, exc)
                    self._store = {}

    def save(self) -> None:
        with self._lock:
            try:
                temp_path = self.cache_file.with_suffix(".tmp")
                with open(temp_path, "w", encoding="utf-8") as f:
                    json.dump(self._store, f, indent=2, ensure_ascii=False)
                temp_path.replace(self.cache_file)
                logger.info(
                    "Saved %d items to cache %s", len(self._store), self.cache_file
                )
            except Exception as exc:
                logger.warning("Failed to save cache %s: %s", self.cache_file, exc)

    def size(self) -> int:
        with self._lock:
            return len(self._store)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            if self.cache_file.exists():
                self.cache_file.unlink()
