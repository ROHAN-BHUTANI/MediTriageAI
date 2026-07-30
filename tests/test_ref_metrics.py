from typing import Any

import pytest

from ref.metrics.base import BaseMetricProvider
from ref.metrics.pipeline import MetricPipeline
from ref.metrics.providers import ClinicalMetricProvider, PerformanceMetricProvider
from ref.metrics.registry import MetricRegistry
from ref.metrics.types import (
    MetricCollection,
    MetricGroup,
    MetricMetadata,
    MetricReport,
    MetricResult,
    MetricValidationError,
)


class MockMetricProvider(BaseMetricProvider):
    def get_metadata(self) -> MetricMetadata:
        return MetricMetadata(
            name="Mock", version="1.0", description="Mock provider", dependencies=[]
        )

    def collect(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {"val": inputs.get("raw_val", 0)}

    def compute(self, data: dict[str, Any]) -> dict[str, Any]:
        return {"computed_val": data["val"] * 2}

    def validate(self, metrics: dict[str, Any]) -> None:
        if metrics["computed_val"] < 0:
            raise MetricValidationError("Must be >= 0")

    def serialize(self, metrics: dict[str, Any]) -> dict[str, Any]:
        return {"computed_val": float(metrics["computed_val"])}


def test_metric_data_structures():
    res = MetricResult(name="accuracy", value=0.95)
    res.validate()

    with pytest.raises(MetricValidationError):
        MetricResult(name="", value=0.95).validate()

    col = MetricCollection(provider_name="Mock", results={"acc": res})
    col.validate()

    grp = MetricGroup(group_name="MockGroup", collections={"Mock": col})
    grp.validate()

    report = MetricReport(experiment_id="EXP_1", groups={"MockGroup": grp})
    report.validate()

    assert hasattr(report, "report_hash")

    d = report.to_dict()
    assert "report_hash" in d

    report2 = MetricReport.from_dict(d)
    assert report2.report_hash == report.report_hash


def test_provider_interface_compliance():
    provider = MockMetricProvider()
    collection = provider.execute_lifecycle({"raw_val": 5})

    assert isinstance(collection, MetricCollection)
    assert "computed_val" in collection.results
    assert collection.results["computed_val"].value == 10.0


def test_registry_integrity_and_config():
    config = {
        "metrics_engine_enabled": True,
        "enable_provider_clinical": True,
        "enable_provider_performance": False,  # Should be skipped
        "enable_group_eval": True,
    }

    registry = MetricRegistry(config)
    registry.discover_providers([ClinicalMetricProvider, PerformanceMetricProvider])

    registry.register("eval", ["Clinical", "Performance"])

    groups = registry.get_all_groups()
    assert "eval" in groups

    providers = registry.get_providers_for_group("eval")
    # Performance is disabled, only Clinical should be here
    assert len(providers) == 1
    assert isinstance(providers[0], ClinicalMetricProvider)


def test_metric_aggregation_pipeline():
    registry = MetricRegistry({})
    registry.discover_providers([MockMetricProvider])
    registry.register("test_group", ["Mock"])

    pipeline = MetricPipeline(registry, experiment_id="EXP_TEST")

    report = pipeline.execute({"raw_val": 20})

    assert isinstance(report, MetricReport)
    assert "test_group" in report.groups
    assert "Mock" in report.groups["test_group"].collections
    assert (
        report.groups["test_group"].collections["Mock"].results["computed_val"].value
        == 40.0
    )


def test_deterministic_ordering():
    res1 = MetricResult(name="b", value=2)
    res2 = MetricResult(name="a", value=1)

    col = MetricCollection(provider_name="P", results={"b": res1, "a": res2})
    d = col.to_dict()

    # Check that 'results' keys are ordered "a" then "b"
    keys = list(d["results"].keys())
    assert keys == ["a", "b"]
