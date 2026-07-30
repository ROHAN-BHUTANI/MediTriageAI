import hashlib
from pathlib import Path
from typing import Any

import yaml


class Config:
    def __init__(self, config_dict: dict[str, Any], raw_yaml: str = ""):
        self._config = config_dict
        self._raw_yaml = raw_yaml

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            raw_yaml = f.read()
        return cls(yaml.safe_load(raw_yaml) or {}, raw_yaml)

    @property
    def random_seed(self) -> int:
        return self._config.get("random_seed", 1337)

    @property
    def splits(self) -> dict[str, float]:
        return self._config.get("splits", {"train": 0.8, "val": 0.1, "test": 0.1})

    @property
    def active_datasets(self) -> list[str]:
        return self._config.get("active_datasets", [])

    @property
    def augmentation(self) -> dict[str, Any]:
        return self._config.get("augmentation", {})

    @property
    def deduplication(self) -> dict[str, Any]:
        return self._config.get("deduplication", {})

    def to_dict(self) -> dict:
        return self._config

    def get_hash(self) -> str:
        return hashlib.sha256(self._raw_yaml.encode("utf-8")).hexdigest()
