# src/transforms/whitespace_variation.py
"""Whitespace Variation transformation plugin.
Adds or removes random whitespace characters (spaces, tabs) within the text.
"""

import random
from typing import Tuple, Dict

from ..transformation_base import TransformationPlugin

class WhitespaceVariation(TransformationPlugin):
    name = "WhitespaceVariation"

    def apply(self, text: str, rng: random.Random) -> Tuple[str, Dict]:
        chars = list(text)
        mods = []
        for i, ch in enumerate(chars):
            if ch == " " and rng.random() < 0.05:
                # Collapse multiple spaces into one.
                mods.append({"position": i, "type": "collapse"})
                chars[i] = ""
            elif ch != " " and rng.random() < 0.03:
                # Insert a space before this character.
                chars.insert(i, " ")
                mods.append({"position": i, "type": "insert_space"})
        transformed = "".join(chars)
        return transformed, {"plugin": self.name, "modifications": mods}
