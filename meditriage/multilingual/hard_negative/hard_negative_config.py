"""Clinical Hard Negative Generator Configuration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class HardNegativeConfig:
    """Configuration for Clinical Hard Negative Generation Engine."""

    # Number of differential hard negatives to generate per input sample
    negatives_per_sample: int = 2

    # Enable strict clinical validation
    strict_validation: bool = True

    # Deterministic generation seed
    random_seed: int = 42

    # Output directory for reports
    output_dir: str = "results/multilingual/hard_negative"

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path | str) -> HardNegativeConfig:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> HardNegativeConfig:
        cfg = cls()
        for k, v in d.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg
