"""OpenAI LLM Provider.

Production-grade provider using the OpenAI Python SDK.
Supports batching, retry with exponential backoff, rate-limit
handling, per-batch checkpointing, and full metadata tracking.

Requires: pip install openai
Requires: OPENAI_API_KEY environment variable
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from typing import Any

from reconstruction.llm import LLMProvider, register_provider, hash_prompt
from reconstruction.llm.resilience import (
    RateLimitError,
    retry_with_backoff,
    validate_generation,
)

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """OpenAI ChatCompletion API provider.

    Args:
        model: OpenAI model identifier.
        api_key: API key (falls back to OPENAI_API_KEY env var).
        temperature: Sampling temperature.
        max_tokens: Max tokens per generation.
        batch_size: Number of completions per batch.
        max_retries: Max retry attempts on transient failure.
        initial_delay: Initial backoff delay in seconds.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        temperature: float = 0.9,
        max_tokens: int = 256,
        batch_size: int = 5,
        max_retries: int = 5,
        initial_delay: float = 1.0,
    ) -> None:
        self._model_name = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._batch_size = batch_size
        self._max_retries = max_retries
        self._initial_delay = initial_delay
        self._client: Any = None
        self._total_prompt_tokens: int = 0
        self._total_completion_tokens: int = 0
        self._total_latency: float = 0.0
        self._total_calls: int = 0

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        try:
            import openai
        except ImportError as e:
            raise ImportError(
                "OpenAIProvider requires `openai`. "
                "Install with: pip install openai"
            ) from e

        if not self._api_key:
            raise ValueError(
                "OpenAIProvider requires OPENAI_API_KEY environment variable "
                "or api_key constructor argument."
            )

        self._client = openai.OpenAI(api_key=self._api_key)
        logger.info("OpenAI client initialised: model=%s", self._model_name)

    def _call_api(self, prompt: str, n: int = 1) -> tuple[list[str], dict]:
        """Make a single ChatCompletion call with retry and backoff.

        Args:
            prompt: The user prompt.
            n: Number of completions to request.

        Returns:
            Tuple of (list_of_texts, metadata_dict).
        """
        self._ensure_client()

        def _do_call() -> tuple[list[str], dict]:
            t0 = time.time()
            try:
                response = self._client.chat.completions.create(
                    model=self._model_name,
                    messages=[
                        {"role": "system", "content": "You are a medical triage assistant generating realistic patient complaints."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                    n=n,
                )
                latency = time.time() - t0
                self._total_latency += latency
                self._total_calls += 1

                texts = []
                for choice in response.choices:
                    text = choice.message.content.strip() if choice.message.content else ""
                    if text:
                        texts.append(text)

                # Token usage
                prompt_tokens = 0
                completion_tokens = 0
                if response.usage:
                    prompt_tokens = response.usage.prompt_tokens or 0
                    completion_tokens = response.usage.completion_tokens or 0
                    self._total_prompt_tokens += prompt_tokens
                    self._total_completion_tokens += completion_tokens

                meta = {
                    "latency_s": round(latency, 3),
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                    "model": self._model_name,
                    "n_requested": n,
                    "n_returned": len(texts),
                    "finish_reasons": [
                        c.finish_reason for c in response.choices
                    ],
                }
                return texts, meta

            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "rate_limit" in err_str or "rate limit" in err_str:
                    retry_after = None
                    if hasattr(e, "headers"):
                        try:
                            retry_after = float(e.headers.get("retry-after", 1))
                        except Exception:
                            pass
                    raise RateLimitError(retry_after) from e
                raise

        return retry_with_backoff(
            _do_call,
            max_retries=self._max_retries,
            initial_delay=self._initial_delay,
        )

    def generate(self, prompt: str, n: int = 1, **kwargs: Any) -> list[str]:
        results: list[str] = []
        remaining = n

        while remaining > 0:
            batch_n = min(remaining, self._batch_size)
            try:
                texts, meta = self._call_api(prompt, n=batch_n)
                results.extend(texts)
                remaining -= batch_n
            except Exception as e:
                logger.warning("OpenAI generation failed for batch (needed %d more): %s", remaining, e)
                break

        return results[:n]

    def validate(self, text: str, department: str) -> bool:
        return validate_generation(text, department)

    def provider_metadata(self) -> dict[str, Any]:
        return {
            "name": "OpenAIProvider",
            "model": self._model_name,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "total_prompt_tokens": self._total_prompt_tokens,
            "total_completion_tokens": self._total_completion_tokens,
            "total_tokens": self._total_prompt_tokens + self._total_completion_tokens,
            "total_latency_s": round(self._total_latency, 3),
            "total_calls": self._total_calls,
            "avg_latency_s": round(self._total_latency / max(self._total_calls, 1), 3),
        }

    @property
    def name(self) -> str:
        return "OpenAIProvider"


register_provider("openai", OpenAIProvider)
