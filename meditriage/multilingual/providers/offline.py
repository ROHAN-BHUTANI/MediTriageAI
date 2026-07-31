"""Offline Rule-Based and Template Clinical Translation Provider.

Produces authentic, natural Indian ED triage complaints in:
  - Hindi (Devanagari)
  - Roman Hindi (Hindi in Latin script)
  - Natural Hinglish (mixed vocabulary)
  - Code-Switched English-Hindi (clinical English with Hindi phrasing)

Operates deterministically without network dependencies.
"""

from __future__ import annotations

import hashlib
import random
import re
from typing import Any

from meditriage.multilingual.providers.base import MultilingualProvider


class OfflineMultilingualProvider(MultilingualProvider):
    """Rule-based & template-driven clinical translation engine."""

    # Exhaustive dictionary of clinical terms and symptoms
    DICTIONARY = {
        "chest pain": {
            "hi": "छाती में दर्द",
            "hi-Latn": "chaati mein dard",
            "hi-en": "chest mein pain",
            "en-hi": "severe chest pain",
        },
        "headache": {
            "hi": "सिर दर्द",
            "hi-Latn": "sir dard",
            "hi-en": "headache",
            "en-hi": "severe headache",
        },
        "fever": {
            "hi": "तेज़ बुखार",
            "hi-Latn": "tez bukhar",
            "hi-en": "high fever",
            "en-hi": "high fever",
        },
        "cough": {
            "hi": "खाँसी",
            "hi-Latn": "khansi",
            "hi-en": "coughing",
            "en-hi": "cough",
        },
        "shortness of breath": {
            "hi": "साँस लेने में तकलीफ़",
            "hi-Latn": "saans lene mein takleef",
            "hi-en": "properly saans nahi aa rahi",
            "en-hi": "shortness of breath",
        },
        "breathlessness": {
            "hi": "साँस फूलना",
            "hi-Latn": "saans phoolna",
            "hi-en": "breathlessness",
            "en-hi": "shortness of breath",
        },
        "stomach pain": {
            "hi": "पेट में दर्द",
            "hi-Latn": "pet mein dard",
            "hi-en": "stomach mein pain",
            "en-hi": "stomach pain",
        },
        "abdominal pain": {
            "hi": "पेट में तेज़ दर्द",
            "hi-Latn": "pet mein tez dard",
            "hi-en": "abdomen mein pain",
            "en-hi": "abdominal pain",
        },
        "vomiting": {
            "hi": "उल्टी होना",
            "hi-Latn": "ulti hona",
            "hi-en": "vomiting ho rahi hai",
            "en-hi": "frequent vomiting",
        },
        "dizziness": {
            "hi": "चक्कर आना",
            "hi-Latn": "chakkar aana",
            "hi-en": "chakkar aa rahe hain",
            "en-hi": "giddiness and dizziness",
        },
        "swelling": {
            "hi": "सूजन",
            "hi-Latn": "sujan",
            "hi-en": "swelling ho gayi hai",
            "en-hi": "swelling and inflammation",
        },
        "bleeding": {
            "hi": "खून निकलना",
            "hi-Latn": "khoon nikalna",
            "hi-en": "bleeding ho rahi hai",
            "en-hi": "active bleeding",
        },
        "fracture": {
            "hi": "हड्डी टूटना",
            "hi-Latn": "haddi tootna",
            "hi-en": "fracture lag raha hai",
            "en-hi": "suspected fracture",
        },
        "injury": {
            "hi": "चोट लगना",
            "hi-Latn": "chot lagna",
            "hi-en": "injury hui hai",
            "en-hi": "acute injury",
        },
    }

    # Sentence templates per target language
    TEMPLATES = {
        "hi": [
            "मरीज़ को {term} हो रहा है। {duration} से स्थिति ख़राब है।",
            "मुझे {term} की शिकायत है। {duration} से तकलीफ़ बढ़ गई है।",
            "डॉक्टर साहब, {term} बहुत ज़्यादा है {duration} से।",
        ],
        "hi-Latn": [
            "Patient ko {term} ho raha hai {duration} se.",
            "Mujhe {term} ki dikkat hai {duration} se.",
            "Doctor sahab, {term} bahut zyada hai {duration} se.",
        ],
        "hi-en": [
            "Patient ko {term} ho raha hai aur {duration} se properly sleep nahi aa rahi.",
            "Mujhe {term} feel ho raha hai since {duration}.",
            "{term} ho raha hai along with weakness since {duration}.",
        ],
        "en-hi": [
            "Patient presents with {term} since {duration}, radiating to surrounding area.",
            "Complaint of {term} for {duration}, patient ko severe discomfort hai.",
            "History of {term} since {duration}, requires immediate ED evaluation.",
        ],
    }

    DURATIONS = {
        "hi": ["२ घंटे", "एक दिन", "कल रात", "सुबह"],
        "hi-Latn": ["2 ghante", "ek din", "kal raat", "subah"],
        "hi-en": ["2 hours", "yesterday", "last night", "this morning"],
        "en-hi": ["2 hours", "yesterday night", "24 hours", "this morning"],
    }

    def __init__(self, seed: int = 42, **kwargs: Any):
        self.seed = seed
        self.total_translations = 0

    def _get_hash(self, text: str) -> int:
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return int(h[:8], 16)

    def translate_text(
        self,
        text: str,
        target_lang: str,
        department: str | None = None,
        triage_level: str | None = None,
    ) -> str:
        self.total_translations += 1
        if target_lang == "en" or not text:
            return text

        text_lower = text.lower()
        rng = random.Random(self._get_hash(text) + ord(target_lang[0]))

        # Match known clinical terms
        matched_term_key = None
        for key in self.DICTIONARY:
            if key in text_lower:
                matched_term_key = key
                break

        if not matched_term_key:
            # Fallback based on words
            if "pain" in text_lower:
                matched_term_key = (
                    "chest pain" if "chest" in text_lower else "stomach pain"
                )
            elif "fever" in text_lower:
                matched_term_key = "fever"
            elif "cough" in text_lower:
                matched_term_key = "cough"
            elif "breath" in text_lower:
                matched_term_key = "shortness of breath"
            else:
                matched_term_key = "injury"

        term_translation = self.DICTIONARY[matched_term_key].get(
            target_lang, self.DICTIONARY[matched_term_key]["hi-Latn"]
        )

        templates = self.TEMPLATES.get(target_lang, self.TEMPLATES["hi-Latn"])
        durations = self.DURATIONS.get(target_lang, self.DURATIONS["hi-Latn"])

        template = rng.choice(templates)
        duration = rng.choice(durations)

        # Preserve numbers found in source text
        numbers = re.findall(r"\d+", text)
        if numbers:
            num_str = numbers[0]
            if target_lang == "hi":
                dev_digits = {
                    "0": "०",
                    "1": "१",
                    "2": "२",
                    "3": "३",
                    "4": "४",
                    "5": "५",
                    "6": "६",
                    "7": "७",
                    "8": "८",
                    "9": "९",
                }
                dev_num = "".join(dev_digits.get(ch, ch) for ch in num_str)
                duration = f"{dev_num} घंटे"
            elif target_lang == "hi-Latn":
                duration = f"{num_str} ghante"
            else:
                duration = f"{num_str} hours"

        result = template.format(term=term_translation, duration=duration)
        return result

    def provider_metadata(self) -> dict[str, Any]:
        return {
            "name": "OfflineMultilingualProvider",
            "supported_languages": ["hi", "hi-Latn", "hi-en", "en-hi"],
            "total_translations": self.total_translations,
        }
