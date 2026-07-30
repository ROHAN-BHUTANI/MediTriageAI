# src/transforms/hinglish_perturbation_plugin.py
"""Hinglish Perturbation transformation plugin.
Wraps the existing ``perturb_text`` function from ``hinglish_perturbation.py``.
"""

import random

from ..hinglish_perturbation import PerturbationResult, perturb_text
from ..transformation_base import TransformationPlugin


class HinglishPerturbation(TransformationPlugin):
    name = "HinglishPerturbation"

    def __init__(self, substitution_rate: float = 0.5):
        self.substitution_rate = substitution_rate

    def apply(self, text: str, rng: random.Random) -> tuple[str, dict]:
        # Use a deterministic seed derived from rng state.
        seed = rng.randint(0, 2**31 - 1)
        result: PerturbationResult = perturb_text(
            text, seed, substitution_rate=self.substitution_rate
        )
        metadata = {
            "plugin": self.name,
            "seed": seed,
            "substitutions": [
                {"original": orig, "replacement": repl, "description": desc}
                for orig, repl, desc in result.substitutions_applied
            ],
        }
        return result.perturbed, metadata
