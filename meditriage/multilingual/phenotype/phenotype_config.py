"""Clinical Phenotype Augmentation Configuration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PhenotypeConfig:
    """Configuration for Clinical Phenotype Augmentation Engine."""

    # Enabled clinical specialties
    enabled_specialties: list[str] = field(
        default_factory=lambda: [
            "Cardiology",
            "Neurology",
            "Respiratory",
            "Orthopedics",
            "Pediatrics",
            "ENT",
            "Emergency Medicine",
            "General Medicine",
        ]
    )

    # Number of phenotype variants to generate per input sample
    variants_per_sample: int = 3

    # Enable strict clinical rule engine checks
    strict_consistency_checking: bool = True

    # Deterministic generation seed
    random_seed: int = 42

    # Output directory for reports
    output_dir: str = "results/multilingual/phenotype"

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path | str) -> PhenotypeConfig:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PhenotypeConfig:
        cfg = cls()
        for k, v in d.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg
