# src/transforms/abbreviation_compression.py
"""Abbreviation Compression transformation plugin.
Compresses full forms back to common abbreviations.
"""

import random

from ..transformation_base import TransformationPlugin

# Inverse mapping of abbreviation expansion.
_ABBREV_COMPRESS_MAP = {
    "blood pressure": ["BP"],
    "heart rate": ["HR"],
    "respiratory rate": ["RR"],
    "oxygen saturation": ["SpO2"],
    "temperature": ["Temp"],
}


class AbbreviationCompression(TransformationPlugin):
    reversible = True
    name = "AbbreviationCompression"

    def __init__(self, compress_map: dict[str, list] | None = None):
        self.compress_map = (
            compress_map if compress_map is not None else _ABBREV_COMPRESS_MAP
        )

    def apply(self, text: str, rng: random.Random) -> tuple[str, dict]:
        transformed = text
        subs = []
        for phrase, abbrevs in self.compress_map.items():
            if phrase.lower() in transformed.lower() and rng.random() < 0.3:
                replacement = rng.choice(abbrevs)
                # Preserve case of first character.
                if phrase[0].isupper():
                    replacement = replacement.upper()
                transformed = transformed.replace(phrase, replacement)
                subs.append(
                    {
                        "original": phrase,
                        "replacement": replacement,
                        "type": "abbreviation_compression",
                    }
                )
        return transformed, {"plugin": self.name, "substitutions": subs}
