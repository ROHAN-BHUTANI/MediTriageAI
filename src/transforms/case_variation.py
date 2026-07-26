# src/transforms/case_variation.py
"""Case Variation transformation plugin.
Randomly changes the case of alphabetic characters (upper, lower, title).
"""

import random
from typing import Tuple, Dict

from ..transformation_base import TransformationPlugin

class CaseVariation(TransformationPlugin):
    reversible = True
    name = "CaseVariation"

    def apply(self, text: str, rng: random.Random) -> Tuple[str, Dict]:
        transformed_chars = []
        changes = []
        for idx, ch in enumerate(text):
            if ch.isalpha() and rng.random() < 0.1:
                new_ch = rng.choice([ch.upper(), ch.lower(), ch.title()])
                transformed_chars.append(new_ch)
                changes.append({"position": idx, "original": ch, "new": new_ch})
            else:
                transformed_chars.append(ch)
        return "".join(transformed_chars), {"plugin": self.name, "changes": changes}
