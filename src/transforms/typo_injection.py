# src/transforms/typo_injection.py
"""Typo Injection transformation plugin.
Introduces common misspellings by swapping adjacent characters or duplicating letters.
"""

import random
from typing import Tuple, Dict

from ..transformation_base import TransformationPlugin

class TypoInjection(TransformationPlugin):
    name = "TypoInjection"

    def apply(self, text: str, rng: random.Random) -> Tuple[str, Dict]:
        words = text.split()
        transformed = []
        subs = []
        for w in words:
            if len(w) > 3 and rng.random() < 0.2:
                # Choose a typo type
                typo_type = rng.choice(["swap", "duplicate"])
                if typo_type == "swap":
                    idx = rng.randint(0, len(w) - 2)
                    typo_word = w[:idx] + w[idx+1] + w[idx] + w[idx+2:]
                else:  # duplicate a character
                    idx = rng.randint(0, len(w) - 1)
                    typo_word = w[:idx] + w[idx] * 2 + w[idx+1:]
                transformed.append(typo_word)
                subs.append({"original": w, "typo": typo_word, "type": "typo_injection"})
            else:
                transformed.append(w)
        return " ".join(transformed), {"plugin": self.name, "substitutions": subs}
