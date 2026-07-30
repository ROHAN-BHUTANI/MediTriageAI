"""
Data structures for the REF Benchmark & Ablation Engine.

Provides strictly validated, deterministic data representations
for experiment comparison and ablation evaluation.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any


class BenchmarkValidationError(Exception):
    """Raised when benchmark/ablation structures fail validation."""


def _deterministic_dict(d: dict[str, Any]) -> dict[str, Any]:
    """Ensure dictionaries are sorted for deterministic serialization."""
    return dict(sorted(d.items()))


@dataclass(frozen=True)
class BenchmarkDefinition:
    """Core tracking for a specific benchmark task."""

    name: str
    target_metric: str
    baseline_experiment_id: str

    def validate(self) -> None:
        if not self.name or not self.target_metric:
            raise BenchmarkValidationError(
                "BenchmarkDefinition must have a name and target metric."
            )

    def to_dict(self) -> dict[str, Any]:
        return _deterministic_dict(asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkDefinition":
        return cls(**data)


@dataclass(frozen=True)
class BenchmarkRun:
    """A single execution run mapped to a benchmark."""

    experiment_id: str
    metrics: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.experiment_id:
            raise BenchmarkValidationError("BenchmarkRun must have an experiment_id.")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["metrics"] = _deterministic_dict(d["metrics"])
        return _deterministic_dict(d)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkRun":
        return cls(**data)


@dataclass(frozen=True)
class BenchmarkComparison:
    """Pairwise or grouped comparison output."""

    comparison_id: str
    base_run: BenchmarkRun
    candidate_runs: list[BenchmarkRun] = field(default_factory=list)
    deltas: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.comparison_id:
            raise BenchmarkValidationError(
                "BenchmarkComparison must have a comparison_id."
            )
        self.base_run.validate()
        for run in self.candidate_runs:
            run.validate()

    def to_dict(self) -> dict[str, Any]:
        d = {
            "comparison_id": self.comparison_id,
            "base_run": self.base_run.to_dict(),
            "candidate_runs": [run.to_dict() for run in self.candidate_runs],
            "deltas": _deterministic_dict(self.deltas),
        }
        return _deterministic_dict(d)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkComparison":
        base_run = BenchmarkRun.from_dict(data.get("base_run", {}))
        candidate_runs = [
            BenchmarkRun.from_dict(r) for r in data.get("candidate_runs", [])
        ]
        return cls(
            comparison_id=data.get("comparison_id", ""),
            base_run=base_run,
            candidate_runs=candidate_runs,
            deltas=data.get("deltas", {}),
        )


@dataclass(frozen=True)
class BenchmarkSuite:
    """A collection of definitions representing a full test suite."""

    suite_name: str
    definitions: list[BenchmarkDefinition] = field(default_factory=list)

    def validate(self) -> None:
        if not self.suite_name:
            raise BenchmarkValidationError("BenchmarkSuite must have a suite_name.")

    def to_dict(self) -> dict[str, Any]:
        d = {
            "suite_name": self.suite_name,
            "definitions": [defn.to_dict() for defn in self.definitions],
        }
        return _deterministic_dict(d)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkSuite":
        definitions = [
            BenchmarkDefinition.from_dict(d) for d in data.get("definitions", [])
        ]
        return cls(suite_name=data.get("suite_name", ""), definitions=definitions)


@dataclass(frozen=True)
class AblationDefinition:
    """Definition for a specific architectural ablation."""

    ablation_name: str
    removed_components: list[str] = field(default_factory=list)

    def validate(self) -> None:
        if not self.ablation_name:
            raise BenchmarkValidationError(
                "AblationDefinition must have an ablation_name."
            )

    def to_dict(self) -> dict[str, Any]:
        return _deterministic_dict(asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AblationDefinition":
        return cls(**data)


@dataclass(frozen=True)
class AblationRun:
    """An experiment run tied to an ablation definition."""

    experiment_id: str
    ablation: AblationDefinition
    metrics: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.experiment_id:
            raise BenchmarkValidationError("AblationRun must have an experiment_id.")
        self.ablation.validate()

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["ablation"] = self.ablation.to_dict()
        d["metrics"] = _deterministic_dict(self.metrics)
        return _deterministic_dict(d)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AblationRun":
        ablation = AblationDefinition.from_dict(data.get("ablation", {}))
        return cls(
            experiment_id=data.get("experiment_id", ""),
            ablation=ablation,
            metrics=data.get("metrics", {}),
        )


@dataclass(frozen=True)
class AblationSummary:
    """Result of an ablation study."""

    study_name: str
    baseline_run: BenchmarkRun
    ablation_runs: list[AblationRun] = field(default_factory=list)
    impact_scores: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.study_name:
            raise BenchmarkValidationError("AblationSummary must have a study_name.")
        self.baseline_run.validate()
        for run in self.ablation_runs:
            run.validate()

    def to_dict(self) -> dict[str, Any]:
        d = {
            "study_name": self.study_name,
            "baseline_run": self.baseline_run.to_dict(),
            "ablation_runs": [run.to_dict() for run in self.ablation_runs],
            "impact_scores": _deterministic_dict(self.impact_scores),
        }
        return _deterministic_dict(d)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AblationSummary":
        baseline_run = BenchmarkRun.from_dict(data.get("baseline_run", {}))
        ablation_runs = [
            AblationRun.from_dict(r) for r in data.get("ablation_runs", [])
        ]
        return cls(
            study_name=data.get("study_name", ""),
            baseline_run=baseline_run,
            ablation_runs=ablation_runs,
            impact_scores=data.get("impact_scores", {}),
        )


@dataclass(frozen=True)
class BenchmarkSummary:
    """The root output of the Benchmark & Ablation Engine."""

    report_id: str
    comparisons: list[BenchmarkComparison] = field(default_factory=list)
    ablations: list[AblationSummary] = field(default_factory=list)
    summary_hash: str = field(init=False)

    def __post_init__(self):
        d = self.to_dict()
        if "summary_hash" in d:
            del d["summary_hash"]
        sorted_json = json.dumps(d, sort_keys=True, separators=(",", ":"))
        hash_val = hashlib.sha256(sorted_json.encode("utf-8")).hexdigest()
        object.__setattr__(self, "summary_hash", hash_val)

    def validate(self) -> None:
        if not self.report_id:
            raise BenchmarkValidationError("BenchmarkSummary must specify report_id.")
        for cmp in self.comparisons:
            cmp.validate()
        for abl in self.ablations:
            abl.validate()

    def to_dict(self) -> dict[str, Any]:
        d = {
            "report_id": self.report_id,
            "comparisons": [cmp.to_dict() for cmp in self.comparisons],
            "ablations": [abl.to_dict() for abl in self.ablations],
        }
        if hasattr(self, "summary_hash"):
            d["summary_hash"] = self.summary_hash
        return _deterministic_dict(d)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkSummary":
        comparisons = [
            BenchmarkComparison.from_dict(c) for c in data.get("comparisons", [])
        ]
        ablations = [AblationSummary.from_dict(a) for a in data.get("ablations", [])]
        obj = cls(
            report_id=data.get("report_id", ""),
            comparisons=comparisons,
            ablations=ablations,
        )
        if "summary_hash" in data:
            object.__setattr__(obj, "summary_hash", data["summary_hash"])
        return obj
