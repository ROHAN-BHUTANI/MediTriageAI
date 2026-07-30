"""
Base provider for the REF Metrics Engine.

Defines the strictly enforced lifecycle for all metric providers:
collect() -> compute() -> validate() -> serialize() -> report().
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

from ref.metrics.types import MetricCollection, MetricMetadata

logger = logging.getLogger(__name__)


class BaseMetricProvider(ABC):
    """
    Abstract base class for all metric computation engines.
    Forces strict decoupling of raw state collection from math computation.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.raw_data: dict[str, Any] = {}
        self.computed_metrics: dict[str, Any] = {}
        self.collection: MetricCollection | None = None

    @abstractmethod
    def get_metadata(self) -> MetricMetadata:
        """Define identity and versioning for this provider."""

    def execute_lifecycle(self, inputs: dict[str, Any]) -> MetricCollection:
        """
        Immutable execution template.
        """
        logger.debug(f"Executing metric provider lifecycle: {self.__class__.__name__}")

        # Stage 1: Collect
        self.raw_data = self.collect(inputs)

        # Stage 2: Compute
        self.computed_metrics = self.compute(self.raw_data)

        # Stage 3: Validate
        self.validate(self.computed_metrics)

        # Stage 4: Serialize
        serialized_results = self.serialize(self.computed_metrics)

        # Stage 5: Report
        self.collection = self.report(serialized_results)

        return self.collection

    @abstractmethod
    def collect(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """
        Extract necessary raw tensors/data from the universal input payload.
        Do NOT perform computation here.
        """

    @abstractmethod
    def compute(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Apply mathematical transformations to generate metrics.
        """

    @abstractmethod
    def validate(self, metrics: dict[str, Any]) -> None:
        """
        Assert bounds, NaNs, or schema compliance on computed metrics.
        Raises ValueError or MetricValidationError if constraints fail.
        """

    @abstractmethod
    def serialize(self, metrics: dict[str, Any]) -> dict[str, Any]:
        """
        Convert complex types (tensors, numpy arrays) into native Python primitives.
        """

    def report(self, serialized_metrics: dict[str, Any]) -> MetricCollection:
        """
        Wrap the primitives into a strict MetricCollection structure.
        """
        from ref.metrics.types import MetricResult

        meta = self.get_metadata()
        results = {}
        for k, v in serialized_metrics.items():
            results[k] = MetricResult(
                name=k,
                value=v,
                context={"provider": meta.name, "version": meta.version},
            )

        collection = MetricCollection(provider_name=meta.name, results=results)
        collection.validate()
        return collection
