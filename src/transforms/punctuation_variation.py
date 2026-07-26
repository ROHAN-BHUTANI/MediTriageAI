# src/transforms/punctuation_variation.py
"""Punctuation Variation transformation plugin.
Randomly inserts, removes, or substitutes punctuation characters.
"""

import random
from typing import Tuple, Dict

from ..transformation_base import TransformationPlugin

_PUNCTUATION = [".", ",", "!", "?", ";", ":"]

class PunctuationVariation(TransformationPlugin):
    reversible = True
    name = "PunctuationVariation"

    def apply(self, text: str, rng: random.Random) -> Tuple[str, Dict]:
        chars = list(text)
        modifications = []
        for i, ch in enumerate(chars):
            if ch.isalnum() and rng.random() < 0.05:
                # Insert punctuation after this character.
                punct = rng.choice(_PUNCTUATION)
                chars.insert(i + 1, punct)
                modifications.append({"position": i + 1, "type": "insert", "punct": punct})
            elif ch in _PUNCTUATION and rng.random() < 0.1:
                # Remove punctuation.
                modifications.append({"position": i, "type": "remove", "punct": ch})
                chars[i] = ""
            elif ch in _PUNCTUATION and rng.random() < 0.1:
                # Substitute punctuation.
                new_punct = rng.choice(_PUNCTUATION)
                modifications.append({"position": i, "type": "replace", "original": ch, "new": new_punct})
                chars[i] = new_punct
        transformed = "".join(chars)
        return transformed, {"plugin": self.name, "modifications": modifications}
