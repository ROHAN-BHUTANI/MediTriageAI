"""
Concrete provenance fingerprint providers for the REF Provenance Framework.

Implements the domain-specific sub-providers derived from BaseFingerprintProvider.
"""

from typing import Any
from ref.provenance.base import BaseFingerprintProvider

class DummyFingerprintMixin:
    """Provides stub implementations for the provider lifecycle to satisfy interfaces."""
    
    def collect(self, context: dict[str, Any]) -> dict[str, Any]:
        return {}
        
    def fingerprint(self, data: dict[str, Any]) -> dict[str, Any]:
        return {"stub": True}
        
    def validate(self, fingerprinted_data: dict[str, Any]) -> None:
        pass
        
    def serialize(self, fingerprinted_data: dict[str, Any]) -> dict[str, Any]:
        return fingerprinted_data


class DatasetFingerprintProvider(DummyFingerprintMixin, BaseFingerprintProvider):
    def get_provider_name(self) -> str:
        return "DatasetFingerprint"

class ConfigurationFingerprintProvider(DummyFingerprintMixin, BaseFingerprintProvider):
    def get_provider_name(self) -> str:
        return "ConfigurationFingerprint"

class CheckpointFingerprintProvider(DummyFingerprintMixin, BaseFingerprintProvider):
    def get_provider_name(self) -> str:
        return "CheckpointFingerprint"

class GitFingerprintProvider(DummyFingerprintMixin, BaseFingerprintProvider):
    def get_provider_name(self) -> str:
        return "GitFingerprint"

class EnvironmentFingerprintProvider(DummyFingerprintMixin, BaseFingerprintProvider):
    def get_provider_name(self) -> str:
        return "EnvironmentFingerprint"

class HardwareFingerprintProvider(DummyFingerprintMixin, BaseFingerprintProvider):
    def get_provider_name(self) -> str:
        return "HardwareFingerprint"

class DependencyFingerprintProvider(DummyFingerprintMixin, BaseFingerprintProvider):
    def get_provider_name(self) -> str:
        return "DependencyFingerprint"

class ExecutionFingerprintProvider(DummyFingerprintMixin, BaseFingerprintProvider):
    def get_provider_name(self) -> str:
        return "ExecutionFingerprint"
