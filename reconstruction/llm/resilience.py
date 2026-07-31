"""LLM Provider Resilience Utilities.

Shared retry, backoff, rate-limit, batching, and checkpoint logic
used by all production LLM providers.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


class RateLimitError(Exception):
    """Raised when the provider signals a rate-limit (HTTP 429 or equivalent)."""

    def __init__(self, retry_after: float | None = None):
        self.retry_after = retry_after
        super().__init__(f"Rate limited. Retry after {retry_after}s" if retry_after else "Rate limited.")


def retry_with_backoff(
    fn: Callable[..., Any],
    *,
    max_retries: int = 5,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay: float = 60.0,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Any:
    """Call `fn()` with exponential backoff on failure.

    Args:
        fn: Zero-argument callable to execute.
        max_retries: Maximum number of retry attempts.
        initial_delay: Initial wait in seconds.
        backoff_factor: Multiplier for each successive retry.
        max_delay: Cap on wait time.
        retryable_exceptions: Exception types that trigger a retry.

    Returns:
        The return value of `fn()` on success.

    Raises:
        The last exception if all retries are exhausted.
    """
    delay = initial_delay
    last_exc: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            return fn()
        except RateLimitError as e:
            last_exc = e
            wait = e.retry_after if e.retry_after else delay
            logger.warning("Rate limited (attempt %d/%d), waiting %.1fs", attempt + 1, max_retries, wait)
            time.sleep(wait)
            delay = min(delay * backoff_factor, max_delay)
        except retryable_exceptions as e:
            last_exc = e
            if attempt == max_retries:
                break
            logger.warning("Retryable error (attempt %d/%d): %s. Waiting %.1fs", attempt + 1, max_retries, e, delay)
            time.sleep(delay)
            delay = min(delay * backoff_factor, max_delay)

    raise last_exc  # type: ignore[misc]


class BatchCheckpointer:
    """Manages batch-level checkpointing for interrupted generation.

    Saves accepted samples to a JSONL file after every successful batch,
    enabling seamless resume.
    """

    def __init__(self, checkpoint_path: Path | str):
        self.path = Path(checkpoint_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._samples: list[dict[str, Any]] = []

        # Load existing checkpoint
        if self.path.exists():
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self._samples.append(json.loads(line))
            logger.info("Resumed %d samples from checkpoint %s", len(self._samples), self.path)

    @property
    def count(self) -> int:
        return len(self._samples)

    @property
    def samples(self) -> list[dict[str, Any]]:
        return list(self._samples)

    def append_batch(self, batch: list[dict[str, Any]]) -> None:
        """Append a batch of samples and flush to disk."""
        with open(self.path, "a", encoding="utf-8") as f:
            for sample in batch:
                f.write(json.dumps(sample, default=str) + "\n")
        self._samples.extend(batch)

    def clear(self) -> None:
        """Remove the checkpoint file."""
        if self.path.exists():
            self.path.unlink()
        self._samples.clear()


def validate_generation(text: str, department: str, min_length: int = 10) -> bool:
    """Common validation logic for generated text.

    Args:
        text: Generated text.
        department: Expected department.
        min_length: Minimum acceptable character count.

    Returns:
        True if the text passes all checks.
    """
    if not text or not isinstance(text, str):
        return False
    text = text.strip()
    if len(text) < min_length:
        return False
    # Reject if it looks like a refusal or meta-response
    refusal_markers = [
        "i cannot", "i can't", "as an ai", "i'm sorry",
        "i am sorry", "i apologize", "not appropriate",
    ]
    text_lower = text.lower()
    if any(marker in text_lower for marker in refusal_markers):
        return False
    # Reject if it's just the department name
    if text_lower.strip() == department.lower().strip():
        return False
    return True
