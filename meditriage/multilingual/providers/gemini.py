"""Google Gemini Provider for Multilingual Clinical Expansion.

Requires: pip install google-generativeai
Requires: GEMINI_API_KEY environment variable
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from meditriage.multilingual.providers.base import MultilingualProvider

logger = logging.getLogger(__name__)


class GeminiMultilingualProvider(MultilingualProvider):
    """Google Gemini API multilingual translation provider."""

    PROMPT_TEMPLATE = """You are a medical triage assistant in an Indian emergency department.
Translate/convert the following patient complaint into natural, authentic {language_name} as spoken in Indian hospitals.

Target Language Specification:
- "hi": Natural Hindi in Devanagari script (e.g. "मेरी छाती में दर्द हो रहा है और साँस लेने में तकलीफ़ है।")
- "hi-Latn": Roman Hindi in Latin script (e.g. "Meri chaati mein dard ho raha hai aur saans lene mein takleef hai.")
- "hi-en": Natural Hinglish mix (e.g. "Chest mein bahut pain ho raha hai aur properly saans nahi aa rahi.")
- "en-hi": Code-switched clinical phrase (e.g. "Patient ko severe chest pain hai with radiation to left arm.")

Requirements:
- Preserve clinical severity, symptoms, and numbers exactly.
- Do NOT sound like a textbook or Google Translate.
- Output ONLY the translated clinical complaint, nothing else.

Original Complaint: {text}
Target Language Code: {target_lang}
Department: {department}

Translated Complaint:"""

    LANG_NAMES = {
        "hi": "Hindi (Devanagari)",
        "hi-Latn": "Roman Hindi (Latin Script)",
        "hi-en": "Natural Hinglish",
        "en-hi": "Code-switched English-Hindi",
    }

    def __init__(
        self,
        model_name: str = "gemini-2.0-flash",
        api_key: str | None = None,
        max_retries: int = 5,
        initial_delay: float = 1.0,
    ):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self._client: Any = None
        self._model: Any = None
        self.total_translations = 0
        self.total_latency = 0.0

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        try:
            import google.generativeai as genai
        except ImportError as e:
            raise ImportError(
                "GeminiMultilingualProvider requires `google-generativeai` package."
            ) from e

        if not self.api_key:
            raise ValueError(
                "GeminiMultilingualProvider requires GEMINI_API_KEY environment variable."
            )

        genai.configure(api_key=self.api_key)
        self._client = genai
        self._model = genai.GenerativeModel(model_name=self.model_name)

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
        lang_name = self.LANG_NAMES.get(target_lang, target_lang)
        prompt = self.PROMPT_TEMPLATE.format(
            language_name=lang_name,
            text=text,
            target_lang=target_lang,
            department=department or "General",
        )

        delay = self.initial_delay
        for attempt in range(self.max_retries + 1):
            try:
                t0 = time.time()
                response = self._model.generate_content(prompt)
                latency = time.time() - t0
                self.total_latency += latency
                self.total_translations += 1

                if response and response.text:
                    return response.text.strip()
                raise ValueError("Empty response from Gemini model.")
            except Exception as exc:
                if attempt == self.max_retries:
                    logger.warning(
                        "Gemini API call failed after %d retries: %s",
                        self.max_retries,
                        exc,
                    )
                    raise
                time.sleep(delay)
                delay *= 2.0

        return text

    def provider_metadata(self) -> dict[str, Any]:
        return {
            "name": "GeminiMultilingualProvider",
            "model_name": self.model_name,
            "total_translations": self.total_translations,
            "total_latency": round(self.total_latency, 3),
        }
