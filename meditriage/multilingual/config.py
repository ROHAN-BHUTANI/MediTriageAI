"""Multilingual Dataset Expansion Configuration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class MultilingualConfig:
    """Configuration for Multilingual Dataset Expansion Engine."""

    # Target language variants
    # en        : Preserved original English
    # hi        : Hindi in Devanagari script
    # hi-Latn   : Roman Hindi (Hindi in Latin script)
    # hi-en     : Natural Hinglish (mixed vocabulary)
    # en-hi     : Code-switched English-Hindi clinical phrasing
    target_languages: list[str] = field(
        default_factory=lambda: ["en", "hi", "hi-Latn", "hi-en", "en-hi"]
    )

    # Provider selection: "offline" | "gemini" | "openai"
    provider: str = "offline"
    model_name: str = "gemini-2.0-flash"

    # Execution settings
    batch_size: int = 50
    num_workers: int = 4
    random_seed: int = 42

    # Resilience & API policy
    max_retries: int = 5
    initial_delay: float = 1.0
    backoff_factor: float = 2.0

    # Caching & Outputs
    cache_dir: str = "results/multilingual_cache"
    output_dir: str = "results/multilingual"
    dataset_path: str = "meditriage/data/processed/dataset.parquet"

    # Quality & Validation thresholds
    min_quality_score: float = 0.8
    preserve_original: bool = True
    strict_validation: bool = True

    # Linguistic Variation Engine integration
    enable_variations: bool = False
    variation_config: dict[str, Any] = field(default_factory=dict)

    # Phenotype Augmentation Engine integration
    enable_phenotype_augmentation: bool = False
    phenotype_config: dict[str, Any] = field(default_factory=dict)

    # Hard Negative Generation Engine integration
    enable_hard_negatives: bool = False
    hard_negative_config: dict[str, Any] = field(default_factory=dict)

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path | str) -> MultilingualConfig:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MultilingualConfig:
        cfg = cls()
        for k, v in d.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg
