"""Clinical Linguistic Variation Engine Configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class VariationConfig:
    """Configuration for Clinical Linguistic Variation Engine."""

    # Enabled variation styles
    enabled_styles: list[str] = field(
        default_factory=lambda: [
            "lexical",
            "syntactic",
            "conversational",
            "ed_triage",
            "physician_note",
            "nurse_intake",
            "abbreviated_notation",
            "formal_documentation",
            "colloquial_indian",
        ]
    )

    # Variation budgets per source sample (max variants per style)
    variation_budgets: dict[str, int] = field(
        default_factory=lambda: {
            "lexical": 2,
            "syntactic": 2,
            "conversational": 2,
            "ed_triage": 2,
            "physician_note": 1,
            "nurse_intake": 1,
            "abbreviated_notation": 1,
            "formal_documentation": 1,
            "colloquial_indian": 2,
        }
    )

    # Total max variants per source record across all styles
    max_variants_per_sample: int = 8

    # Quality & Similarity thresholds
    min_semantic_similarity: float = 0.70
    strict_semantic_preservation: bool = True
    random_seed: int = 42

    # Output directory for reports
    output_dir: str = "results/multilingual/variation"

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path | str) -> VariationConfig:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> VariationConfig:
        cfg = cls()
        for k, v in d.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg
