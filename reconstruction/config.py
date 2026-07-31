"""Dataset Reconstruction Engine – Global Configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class ReconstructionConfig:
    """All configurable parameters for the reconstruction engine."""

    # ── Core ──────────────────────────────────────────────────
    target_class_size: int = 25_000
    random_seed: int = 42
    output_directory: str = "results/reconstruction"

    # ── Dataset ───────────────────────────────────────────────
    dataset_path: str = "meditriage/data/processed/dataset.parquet"
    required_columns: list[str] = field(
        default_factory=lambda: ["raw_text", "department"]
    )

    # ── Cleaning (Stage 2) ────────────────────────────────────
    min_text_length: int = 3
    max_text_length: int = 50_000

    # ── Clustering (Stage 3) ──────────────────────────────────
    clustering_algorithm: str = "minibatch_kmeans"  # minibatch_kmeans | hdbscan
    embedding_model: str = "tfidf"  # tfidf | sentence_transformer
    sentence_transformer_name: str = "all-MiniLM-L6-v2"
    max_clusters_per_department: int = 50
    min_cluster_size: int = 5
    tfidf_max_features: int = 10_000

    # ── Diversity Scoring (Stage 4) ───────────────────────────
    diversity_weights: dict[str, float] = field(
        default_factory=lambda: {
            "lexical": 0.25,
            "semantic": 0.25,
            "symptom": 0.20,
            "language": 0.15,
            "text_length": 0.15,
        }
    )

    # ── Undersampling (Stage 5) ───────────────────────────────
    similarity_threshold: float = 0.95

    # ── Augmentation (Stage 6 – future) ───────────────────────
    augmentation_languages: list[str] = field(
        default_factory=lambda: [
            "english",
            "hindi",
            "roman_hindi",
            "hinglish",
            "broken_english",
            "broken_hinglish",
            "sms_shorthand",
        ]
    )
    augmentation_probability: float = 0.5
    augmentation_min_class_size: int = 500

    # ── LLM Generation (Stage 7) ─────────────────────────────
    llm_provider: str = "gemini"  # gemini | openai | offline
    llm_model: str = "gemini-2.0-flash"
    llm_temperature: float = 0.9
    llm_max_output_tokens: int = 256
    llm_batch_size: int = 5
    llm_max_retries: int = 5
    llm_initial_delay: float = 1.0
    llm_min_class_threshold: int = 500

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path | str) -> "ReconstructionConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def from_overrides(cls, overrides: dict[str, Any]) -> "ReconstructionConfig":
        cfg = cls()
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg
