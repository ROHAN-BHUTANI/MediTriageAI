# src/transforms/clinical_shorthand.py
"""Clinical Shorthand transformation plugin.
Converts full clinical phrasing to common shorthand tokens.
"""

import random

from ..transformation_base import TransformationPlugin

_SHORTHAND_MAP = {
    "patient presents with": ["pt c/o"],
    "history of": ["hx"],
    "no significant findings": ["NSF"],
    "blood pressure": ["BP"],
    "heart rate": ["HR"],
}


class ClinicalShorthand(TransformationPlugin):
    reversible = True
    name = "ClinicalShorthand"

    def __init__(self, shorthand_map: dict[str, list] = None):
        self.shorthand_map = (
            shorthand_map if shorthand_map is not None else _SHORTHAND_MAP
        )

    def apply(self, text: str, rng: random.Random) -> tuple[str, dict]:
        # Simple token replacement based on substring matching.
        transformed = text
        subs = []
        for phrase, replacements in self.shorthand_map.items():
            if phrase.lower() in transformed.lower() and rng.random() < 0.3:
                replacement = rng.choice(replacements)
                # Preserve case of first character.
                if phrase[0].isupper():
                    replacement = replacement.capitalize()
                transformed = transformed.replace(phrase, replacement)
                subs.append(
                    {
                        "original": phrase,
                        "replacement": replacement,
                        "type": "clinical_shorthand",
                    }
                )
        return transformed, {"plugin": self.name, "substitutions": subs}
