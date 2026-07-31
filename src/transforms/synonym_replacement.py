# src/transforms/synonym_replacement.py
"""Synonym Replacement transformation plugin.

A simple deterministic synonym mapper for medical terms. The mapping is hard‑coded
for demonstration but can be extended via a JSON/YAML file.
"""

import random

from ..transformation_base import TransformationPlugin

# Minimal synonym dictionary – in a real system this would be comprehensive.
_SYNONYM_MAP = {
    "vomiting": ["throwing up", "emesis"],
    "fever": ["pyrexia", "high temperature"],
    "headache": ["cephalgia", "migraine"],
    "pain": ["ache", "discomfort"],
    "blood pressure": ["BP", "blood pressure reading"],
}


class SynonymReplacement(TransformationPlugin):
    reversible = True
    name = "SynonymReplacement"

    def __init__(self, synonym_map: dict[str, list] | None = None):
        # Allow injection of a custom map for testing.
        self.synonym_map = synonym_map if synonym_map is not None else _SYNONYM_MAP

    def apply(self, text: str, rng: random.Random) -> tuple[str, dict]:
        words = text.split()
        transformed_words = []
        substitutions = []
        for w in words:
            key = w.lower().strip(".,;!")
            if key in self.synonym_map and rng.random() < 0.3:
                replacement = rng.choice(self.synonym_map[key])
                # Preserve original casing.
                if w[0].isupper():
                    replacement = replacement.capitalize()
                transformed_words.append(replacement)
                substitutions.append(
                    {"original": w, "replacement": replacement, "type": "synonym"}
                )
            else:
                transformed_words.append(w)
        transformed_text = " ".join(transformed_words)
        metadata = {
            "plugin": self.name,
            "substitutions": substitutions,
        }
        return transformed_text, metadata
