"""Deterministic Offline LLM Provider.

Template-based generation that requires no API keys or network.
Used as the default fallback and for testing.  Produces clinically
diverse samples by combining seed texts with structural templates
across multiple languages and demographics.
"""

from __future__ import annotations

import random
import re
from typing import Any

from reconstruction.llm import LLMProvider, register_provider

_DEMOGRAPHICS = [
    "A 25-year-old male", "A 45-year-old female", "A 7-year-old child",
    "A 60-year-old man", "A 32-year-old woman", "A 70-year-old elderly patient",
    "An 18-year-old college student", "A 50-year-old construction worker",
    "A 38-year-old pregnant woman", "A 55-year-old diabetic patient",
    "A teenager", "An elderly woman",
]

_CONTEXTS = [
    "presents to the emergency room",
    "visits the clinic",
    "is brought in by family members",
    "arrives by ambulance",
    "calls the helpline",
    "walks into the hospital",
    "is referred by a local doctor",
    "comes for a follow-up",
]

_CONNECTORS = [
    "complaining of", "reporting", "with symptoms of",
    "experiencing", "suffering from", "having",
]

_DURATIONS = [
    "since this morning", "for 2 days", "for a week",
    "since yesterday", "for the past 3 hours", "intermittently for a month",
    "suddenly since last night", "gradually worsening over 5 days",
]

_HINDI_TEMPLATES = [
    "Mujhe {symptom} ho raha hai {duration}",
    "{duration} se {symptom} hai, bahut takleef hai",
    "Doctor sahab, {symptom} bahut zyada hai",
    "Mera {symptom} theek nahi ho raha, {duration}",
]

_HINGLISH_TEMPLATES = [
    "I am having {symptom} {duration}, please help",
    "Mujhe {symptom} hai since {duration}, kya karu?",
    "Doctor, {symptom} ho gaya hai, {duration} se",
]


class OfflineProvider(LLMProvider):
    """Template-based offline generation for testing and fallback."""

    def generate(self, prompt: str, n: int = 1, **kwargs: Any) -> list[str]:
        seed = kwargs.get("seed", 42)
        rng = random.Random(seed)
        results = []

        # Extract symptom hint from prompt
        symptom_match = re.search(r"symptoms?:\s*(.+?)(?:\n|$)", prompt, re.IGNORECASE)
        symptom = symptom_match.group(1).strip() if symptom_match else "general discomfort"

        for i in range(n):
            rng_i = random.Random(seed + i)
            lang_choice = rng_i.randint(0, 2)

            if lang_choice == 0:
                # English
                demo = rng_i.choice(_DEMOGRAPHICS)
                context = rng_i.choice(_CONTEXTS)
                connector = rng_i.choice(_CONNECTORS)
                duration = rng_i.choice(_DURATIONS)
                text = f"{demo} {context} {connector} {symptom} {duration}."
            elif lang_choice == 1:
                # Hindi
                template = rng_i.choice(_HINDI_TEMPLATES)
                duration = rng_i.choice(["kal se", "do din se", "ek hafte se", "subah se"])
                text = template.format(symptom=symptom, duration=duration)
            else:
                # Hinglish
                template = rng_i.choice(_HINGLISH_TEMPLATES)
                duration = rng_i.choice(["yesterday", "2 din", "morning se", "last week"])
                text = template.format(symptom=symptom, duration=duration)

            results.append(text)

        return results

    def validate(self, text: str, department: str) -> bool:
        if not text or len(text) < 5:
            return False
        return True

    def provider_metadata(self) -> dict[str, Any]:
        return {"name": "OfflineProvider", "version": "1.0", "type": "template-based"}


register_provider("offline", OfflineProvider)
