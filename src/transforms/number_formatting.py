# src/transforms/number_formatting.py
"""Number Formatting transformation plugin.
Applies deterministic formatting to numeric tokens (e.g., zero‑padding, thousand separators).
"""

import random
import re
from typing import Tuple, Dict

from ..transformation_base import TransformationPlugin

_NUMBER_REGEX = re.compile(r"\b\d+\b")

class NumberFormatting(TransformationPlugin):
    reversible = True
    name = "NumberFormatting"

    def apply(self, text: str, rng: random.Random) -> Tuple[str, Dict]:
        def _replace(match: re.Match[str]) -> str:
            num_str = match.group(0)
            if rng.random() < 0.4:
                # Zero‑pad to 2 digits.
                return num_str.zfill(2)
            if rng.random() < 0.2:
                # Add thousand separator if length > 3.
                if len(num_str) > 3:
                    return f"{int(num_str):,}"
            return num_str
        transformed = _NUMBER_REGEX.sub(_replace, text)
        return transformed, {"plugin": self.name, "action": "number_formatting"}
