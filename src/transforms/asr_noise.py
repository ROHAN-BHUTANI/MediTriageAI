# src/transforms/asr_noise.py
"""ASR Noise transformation plugin.
Simulates automatic speech recognition errors by substituting common phonetic mistakes.
"""

import random
from typing import Tuple, Dict

from ..transformation_base import TransformationPlugin

# Simple mapping of words to typical ASR errors.
_ASR_MAP = {
    "pain": ["pane", "pan"],
    "blood": ["bloed", "blod"],
    "pressure": ["presure", "preasure"],
    "temperature": ["tempereture", "temperatur"],
    "heart": ["hart", "hart"],
}

class ASRNoise(TransformationPlugin):
    reversible = True
    name = "ASRNoise"

    def __init__(self, asr_map: Dict[str, list] = None):
        self.asr_map = asr_map if asr_map is not None else _ASR_MAP

    def apply(self, text: str, rng: random.Random) -> Tuple[str, Dict]:
        words = text.split()
        transformed = []
        subs = []
        for w in words:
            key = w.lower().strip('.,;!')
            if key in self.asr_map and rng.random() < 0.25:
                replacement = rng.choice(self.asr_map[key])
                # Preserve original case.
                if w[0].isupper():
                    replacement = replacement.capitalize()
                transformed.append(replacement)
                subs.append({"original": w, "replacement": replacement, "type": "asr_noise"})
            else:
                transformed.append(w)
        return " ".join(transformed), {"plugin": self.name, "substitutions": subs}
