# src/transforms/duration_variation.py
"""Duration Variation transformation plugin.
Alters temporal expressions (e.g., "2 days", "3 weeks") to generate variants.
"""

import random
import re

from ..transformation_base import TransformationPlugin

# Simple regex for number + unit.
_DURATION_REGEX = re.compile(
    r"\b(\d+)\s*(seconds?|minutes?|hours?|days?|weeks?|months?|years?)\b", re.IGNORECASE
)

_VARIANTS = {
    "seconds": ["secs", "s"],
    "minutes": ["mins", "m"],
    "hours": ["hrs", "h"],
    "days": ["d"],
    "weeks": ["w"],
    "months": ["mos"],
    "years": ["yrs", "y"],
}


class DurationVariation(TransformationPlugin):
    name = "DurationVariation"

    def apply(self, text: str, rng: random.Random) -> tuple[str, dict]:
        def _replace(match: re.Match[str]) -> str:
            number, unit = match.groups()
            unit = unit.lower()
            base = unit.rstrip("s")
            if base in _VARIANTS and rng.random() < 0.4:
                variant = rng.choice(_VARIANTS[base])
                return f"{number} {variant}"
            return match.group(0)

        transformed = _DURATION_REGEX.sub(_replace, text)
        return transformed, {"plugin": self.name, "action": "duration_variation"}
