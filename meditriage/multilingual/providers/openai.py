"""OpenAI Provider for Multilingual Clinical Expansion.

Requires: pip install openai
Requires: OPENAI_API_KEY environment variable
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from meditriage.multilingual.providers.base import MultilingualProvider

logger = logging.getLogger(__name__)


class OpenAIMultilingualProvider(MultilingualProvider):
    """OpenAI API multilingual translation provider."""

    SYSTEM_PROMPT = "You are a clinical translation assistant specializing in Indian Emergency Department triage conversations."

    USER_PROMPT = """Translate/convert the following clinical complaint into authentic {target_lang} as spoken in Indian emergency departments.

Target Specifications:
- "hi": Natural Hindi in Devanagari script.
- "hi-Latn": Roman Hindi in Latin script.
- "hi-en": Natural Hinglish mix of English and Hindi words.
- "en-hi": Code-switched clinical phrasing.

Preserve all symptoms, severity, and numbers.
Output ONLY the translated complaint text.

Complaint: {text}
Department: {department}"""

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        api_key: str | None = None,
        max_retries: int = 5,
        initial_delay: float = 1.0,
    ):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self._client: Any = None
        self.total_translations = 0
        self.total_latency = 0.0

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        try:
            import openai
        except ImportError as e:
            raise ImportError(
                "OpenAIMultilingualProvider requires `openai` package."
            ) from e

        if not self.api_key:
            raise ValueError(
                "OpenAIMultilingualProvider requires OPENAI_API_KEY environment variable."
            )

        self._client = openai.OpenAI(api_key=self.api_key)

    def translate_text(
        self,
        text: str,
        target_lang: str,
        department: str | None = None,
        triage_level: str | None = None,
    ) -> str:
        if target_lang == "en" or not text:
            return text

        self._ensure_client()
        prompt = self.USER_PROMPT.format(
            target_lang=target_lang,
            text=text,
            department=department or "General",
        )

        delay = self.initial_delay
        for attempt in range(self.max_retries + 1):
            try:
                t0 = time.time()
                response = self._client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.7,
                )
                latency = time.time() - t0
                self.total_latency += latency
                self.total_translations += 1

                content = response.choices[0].message.content
                if content:
                    return content.strip()
                raise ValueError("Empty completion from OpenAI model.")
            except Exception as exc:
                if attempt == self.max_retries:
                    logger.warning(
                        "OpenAI API call failed after %d retries: %s",
                        self.max_retries,
                        exc,
                    )
                    raise
                time.sleep(delay)
                delay *= 2.0

        return text

    def provider_metadata(self) -> dict[str, Any]:
        return {
            "name": "OpenAIMultilingualProvider",
            "model_name": self.model_name,
            "total_translations": self.total_translations,
            "total_latency": round(self.total_latency, 3),
        }
