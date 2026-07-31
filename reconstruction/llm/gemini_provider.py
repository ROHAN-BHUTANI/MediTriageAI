"""Gemini LLM Provider.

Production-grade provider using the Google Generative AI SDK.
Supports batching, retry with exponential backoff, rate-limit
handling, per-batch checkpointing, and full metadata tracking.

Requires: pip install google-generativeai
Requires: GEMINI_API_KEY environment variable
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


class GeminiProvider(LLMProvider):
    """Google Gemini API provider.

    Args:
        model: Gemini model identifier.
        api_key: API key (falls back to GEMINI_API_KEY env var).
        temperature: Sampling temperature.
        max_output_tokens: Max tokens per generation.
        batch_size: Number of completions per API call.
        max_retries: Max retry attempts on transient failure.
        initial_delay: Initial backoff delay in seconds.
    """

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        api_key: str | None = None,
        temperature: float = 0.9,
        max_output_tokens: int = 256,
        batch_size: int = 5,
        max_retries: int = 5,
        initial_delay: float = 1.0,
    ) -> None:
        self._model_name = model
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        self._batch_size = batch_size
        self._max_retries = max_retries
        self._initial_delay = initial_delay
        self._client: Any = None
        self._model: Any = None
        self._total_tokens: int = 0
        self._total_latency: float = 0.0
        self._total_calls: int = 0

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        try:
            import google.generativeai as genai
        except ImportError as e:
            raise ImportError(
                "GeminiProvider requires `google-generativeai`. "
                "Install with: pip install google-generativeai"
            ) from e

        if not self._api_key:
            raise ValueError(
                "GeminiProvider requires GEMINI_API_KEY environment variable "
                "or api_key constructor argument."
            )

        genai.configure(api_key=self._api_key)
        self._client = genai
        self._model = genai.GenerativeModel(
            model_name=self._model_name,
            generation_config={
                "temperature": self._temperature,
                "max_output_tokens": self._max_output_tokens,
            },
        )
        logger.info("Gemini client initialised: model=%s", self._model_name)

    def _call_api(self, prompt: str) -> tuple[str, dict]:
        """Make a single API call with retry and backoff.

        Returns:
            Tuple of (generated_text, metadata_dict).
        """
        self._ensure_client()

        def _do_call() -> tuple[str, dict]:
            t0 = time.time()
            try:
                response = self._model.generate_content(prompt)
                latency = time.time() - t0
                self._total_latency += latency
                self._total_calls += 1

                text = ""
                if response and response.text:
                    text = response.text.strip()

                # Extract token usage
                tokens_used = 0
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    tokens_used = getattr(response.usage_metadata, "total_token_count", 0)
                    self._total_tokens += tokens_used

                meta = {
                    "latency_s": round(latency, 3),
                    "tokens_used": tokens_used,
                    "model": self._model_name,
                    "finish_reason": getattr(response.candidates[0], "finish_reason", None)
                    if response and response.candidates else None,
                }
                return text, meta

            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "rate" in err_str or "quota" in err_str:
                    # Try to extract retry-after
                    retry_after = None
                    if "retry-after" in err_str:
                        try:
                            import re
                            m = re.search(r"retry-after[:\s]+(\d+)", err_str)
                            if m:
                                retry_after = float(m.group(1))
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
        for i in range(n):
            try:
                text, meta = self._call_api(prompt)
                if text:
                    results.append(text)
            except Exception as e:
                logger.warning("Gemini generation failed for sample %d: %s", i, e)
                continue
        return results

    def validate(self, text: str, department: str) -> bool:
        return validate_generation(text, department)

    def provider_metadata(self) -> dict[str, Any]:
        return {
            "name": "GeminiProvider",
            "model": self._model_name,
            "temperature": self._temperature,
            "max_output_tokens": self._max_output_tokens,
            "total_tokens": self._total_tokens,
            "total_latency_s": round(self._total_latency, 3),
            "total_calls": self._total_calls,
            "avg_latency_s": round(self._total_latency / max(self._total_calls, 1), 3),
        }

    @property
    def name(self) -> str:
        return "GeminiProvider"


register_provider("gemini", GeminiProvider)
