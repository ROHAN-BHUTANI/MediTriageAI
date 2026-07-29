"""
Base provider for the REF Provenance Framework.

Defines the strictly enforced lifecycle for all provenance fingerprint providers:
collect() -> fingerprint() -> validate() -> serialize() -> report().
"""

from abc import ABC, abstractmethod
from typing import Any
import logging

from ref.provenance.types import ExperimentFingerprint

logger = logging.getLogger(__name__)

class BaseFingerprintProvider(ABC):
    """
    Abstract base class for all provenance tracking engines.
    Forces strict decoupling of raw state collection from serialization and reporting.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.raw_data: dict[str, Any] = {}
        self.fingerprinted_data: dict[str, Any] = {}
        self.serialized_data: dict[str, Any] = {}
        self.experiment_fingerprint: ExperimentFingerprint | None = None

    @abstractmethod
    def get_provider_name(self) -> str:
        """Define identity for this provider."""

    def execute_lifecycle(self, context: dict[str, Any]) -> ExperimentFingerprint:
        """
        Immutable execution template.
        """
        logger.debug(f"Executing provenance lifecycle: {self.__class__.__name__}")
        
        # Stage 1: Collect
        self.raw_data = self.collect(context)
        
        # Stage 2: Fingerprint
        self.fingerprinted_data = self.fingerprint(self.raw_data)
        
        # Stage 3: Validate
        self.validate(self.fingerprinted_data)
        
        # Stage 4: Serialize
        self.serialized_data = self.serialize(self.fingerprinted_data)
        
        # Stage 5: Report
        self.experiment_fingerprint = self.report(self.serialized_data)
        
        return self.experiment_fingerprint

    @abstractmethod
    def collect(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Extract necessary raw system state or configuration data.
        """

    @abstractmethod
    def fingerprint(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Apply deterministic transformations or struct mappings.
        """

    @abstractmethod
    def validate(self, fingerprinted_data: dict[str, Any]) -> None:
        """
        Assert completeness of the fetched data.
        Raises ValueError or ProvenanceValidationError if constraints fail.
        """

    @abstractmethod
    def serialize(self, fingerprinted_data: dict[str, Any]) -> dict[str, Any]:
        """
        Convert instances down into JSON-ready dictionary payloads.
        """

    def report(self, serialized_data: dict[str, Any]) -> ExperimentFingerprint:
        """
        Wrap the primitives into a strict ExperimentFingerprint structure.
        """
        name = self.get_provider_name()
        
        fingerprint = ExperimentFingerprint(
            provider_name=name,
            payload=serialized_data
        )
        fingerprint.validate()
        return fingerprint
