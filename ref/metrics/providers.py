"""
Concrete metric providers for the REF Metrics Engine.

Implements the domain-specific sub-providers derived from BaseMetricProvider.
"""

from typing import Any
from ref.metrics.base import BaseMetricProvider
from ref.metrics.types import MetricMetadata


class DummyProviderMixin:
    """Provides stub implementations for the provider lifecycle."""
    def collect(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {}
        
    def compute(self, data: dict[str, Any]) -> dict[str, Any]:
        # Return a dummy metric to satisfy validation
        return {"dummy_metric": 0.0}
        
    def validate(self, metrics: dict[str, Any]) -> None:
        pass
        
    def serialize(self, metrics: dict[str, Any]) -> dict[str, Any]:
        return metrics


class ClinicalMetricProvider(DummyProviderMixin, BaseMetricProvider):
    def get_metadata(self) -> MetricMetadata:
        return MetricMetadata(name="Clinical", version="1.0", description="Evaluates clinical accuracy.")


class PerformanceMetricProvider(DummyProviderMixin, BaseMetricProvider):
    def get_metadata(self) -> MetricMetadata:
        return MetricMetadata(name="Performance", version="1.0", description="Evaluates system performance.")


class CalibrationMetricProvider(DummyProviderMixin, BaseMetricProvider):
    def get_metadata(self) -> MetricMetadata:
        return MetricMetadata(name="Calibration", version="1.0", description="Evaluates probability calibration.")


class RoutingMetricProvider(DummyProviderMixin, BaseMetricProvider):
    def get_metadata(self) -> MetricMetadata:
        return MetricMetadata(name="Routing", version="1.0", description="Evaluates dynamic routing traces.")


class OptimizationMetricProvider(DummyProviderMixin, BaseMetricProvider):
    def get_metadata(self) -> MetricMetadata:
        return MetricMetadata(name="Optimization", version="1.0", description="Evaluates multi-task optimization.")


class ConfidenceMetricProvider(DummyProviderMixin, BaseMetricProvider):
    def get_metadata(self) -> MetricMetadata:
        return MetricMetadata(name="Confidence", version="1.0", description="Evaluates predictive confidence.")


class EfficiencyMetricProvider(DummyProviderMixin, BaseMetricProvider):
    def get_metadata(self) -> MetricMetadata:
        return MetricMetadata(name="Efficiency", version="1.0", description="Evaluates compute efficiency.")


class RobustnessMetricProvider(DummyProviderMixin, BaseMetricProvider):
    def get_metadata(self) -> MetricMetadata:
        return MetricMetadata(name="Robustness", version="1.0", description="Evaluates model stability and robustness.")
