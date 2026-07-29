"""
Data structures for the REF Reproducibility & Provenance Framework.

Provides strictly validated, deterministic data representations
for the provenance execution lifecycle.
"""

from dataclasses import dataclass, field, asdict
from typing import Any
import json
import hashlib

class ProvenanceValidationError(Exception):
    """Raised when provenance structures fail validation."""

def _deterministic_dict(d: dict[str, Any]) -> dict[str, Any]:
    """Ensure dictionaries are sorted for deterministic serialization."""
    return dict(sorted(d.items()))

@dataclass(frozen=True)
class ExecutionEnvironment:
    """Core tracking for the execution context."""
    timestamp: str
    seed: int
    ref_version: str
    repository_version: str

    def validate(self) -> None:
        if not self.timestamp:
            raise ProvenanceValidationError("ExecutionEnvironment must have a timestamp.")

    def to_dict(self) -> dict[str, Any]:
        return _deterministic_dict(asdict(self))
        
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExecutionEnvironment":
        return cls(**data)

@dataclass(frozen=True)
class DatasetFingerprint:
    """Checksums and versions for datasets."""
    dataset_name: str
    checksum: str
    version: str

    def validate(self) -> None:
        if not self.dataset_name or not self.checksum:
            raise ProvenanceValidationError("DatasetFingerprint must have name and checksum.")

    def to_dict(self) -> dict[str, Any]:
        return _deterministic_dict(asdict(self))
        
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetFingerprint":
        return cls(**data)

@dataclass(frozen=True)
class ConfigurationFingerprint:
    """Hashed hyperparameter state."""
    config_hash: str
    overrides: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.config_hash:
            raise ProvenanceValidationError("ConfigurationFingerprint must have config_hash.")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["overrides"] = _deterministic_dict(d["overrides"])
        return _deterministic_dict(d)
        
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConfigurationFingerprint":
        return cls(**data)

@dataclass(frozen=True)
class CheckpointFingerprint:
    """Hash and URI for loaded weights."""
    checkpoint_uri: str
    checkpoint_hash: str

    def validate(self) -> None:
        pass

    def to_dict(self) -> dict[str, Any]:
        return _deterministic_dict(asdict(self))
        
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CheckpointFingerprint":
        return cls(**data)

@dataclass(frozen=True)
class HardwareProfile:
    """Machine characteristics."""
    gpu_model: str
    driver_version: str
    cuda_version: str
    cudnn_version: str
    os_info: str

    def validate(self) -> None:
        pass

    def to_dict(self) -> dict[str, Any]:
        return _deterministic_dict(asdict(self))
        
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HardwareProfile":
        return cls(**data)

@dataclass(frozen=True)
class SoftwareProfile:
    """Library and execution versions."""
    python_version: str
    git_commit: str
    git_dirty_flag: bool
    dependency_versions: dict[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        pass

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["dependency_versions"] = _deterministic_dict(d["dependency_versions"])
        return _deterministic_dict(d)
        
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SoftwareProfile":
        return cls(**data)

@dataclass(frozen=True)
class ExperimentFingerprint:
    """A wrapper for a single provider's output."""
    provider_name: str
    payload: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.provider_name:
            raise ProvenanceValidationError("ExperimentFingerprint must have a provider_name.")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["payload"] = _deterministic_dict(d["payload"])
        return _deterministic_dict(d)
        
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentFingerprint":
        return cls(**data)

@dataclass(frozen=True)
class ProvenanceManifest:
    """The root output of the Provenance Framework."""
    experiment_id: str
    fingerprints: dict[str, ExperimentFingerprint] = field(default_factory=dict)
    manifest_hash: str = field(init=False)

    def __post_init__(self):
        d = self.to_dict()
        if "manifest_hash" in d:
            del d["manifest_hash"]
        sorted_json = json.dumps(d, sort_keys=True, separators=(',', ':'))
        hash_val = hashlib.sha256(sorted_json.encode('utf-8')).hexdigest()
        object.__setattr__(self, "manifest_hash", hash_val)

    def validate(self) -> None:
        if not self.experiment_id:
            raise ProvenanceValidationError("ProvenanceManifest must specify experiment_id.")
        for k, v in self.fingerprints.items():
            if not isinstance(v, ExperimentFingerprint):
                raise ProvenanceValidationError(f"Key {k} must map to an ExperimentFingerprint.")
            v.validate()

    def to_dict(self) -> dict[str, Any]:
        d = {
            "experiment_id": self.experiment_id,
            "fingerprints": _deterministic_dict({k: v.to_dict() for k, v in self.fingerprints.items()})
        }
        if hasattr(self, "manifest_hash"):
            d["manifest_hash"] = getattr(self, "manifest_hash")
        return _deterministic_dict(d)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProvenanceManifest":
        fingerprints = data.get("fingerprints", {})
        obj = cls(
            experiment_id=data.get("experiment_id", "unknown"),
            fingerprints={k: ExperimentFingerprint.from_dict(v) for k, v in fingerprints.items()}
        )
        if "manifest_hash" in data:
            object.__setattr__(obj, "manifest_hash", data["manifest_hash"])
        return obj
