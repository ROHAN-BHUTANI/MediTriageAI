# src/transforms/abbreviation_expansion.py
"""Abbreviation Expansion transformation plugin.
Expands common medical abbreviations to their full forms.
"""

import random

from ..transformation_base import TransformationPlugin

_ABBREV_MAP = {
    "BP": ["blood pressure"],
    "HR": ["heart rate"],
    "RR": ["respiratory rate"],
    "SpO2": ["oxygen saturation"],
    "Temp": ["temperature"],
}


class AbbreviationExpansion(TransformationPlugin):
    reversible = True
    name = "AbbreviationExpansion"

    def __init__(self, abbrev_map: dict[str, list] = None):
        self.abbrev_map = abbrev_map if abbrev_map is not None else _ABBREV_MAP

    def apply(self, text: str, rng: random.Random) -> tuple[str, dict]:
        words = text.split()
        transformed = []
        subs = []
        for w in words:
            key = w.strip(".,;!")
            if key in self.abbrev_map and rng.random() < 0.4:
                replacement = rng.choice(self.abbrev_map[key])
                transformed.append(replacement)
                subs.append(
                    {
                        "original": w,
                        "replacement": replacement,
                        "type": "abbreviation_expansion",
                    }
                )
            else:
                transformed.append(w)
        return " ".join(transformed), {"plugin": self.name, "substitutions": subs}
