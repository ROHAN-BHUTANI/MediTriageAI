import json
from unittest.mock import MagicMock

import pandas as pd
import torch

from src.calibration import Calibrator
from src.evaluation import EvaluationExporter, generate_training_report
from src.explainability import ExplainabilityRegistry


def test_prediction_exports(tmp_path):
    exporter = EvaluationExporter(str(tmp_path))

    spec_logits = torch.tensor([[10.0, 0.0], [0.0, 10.0]])
    sev_logits = torch.tensor([[10.0, 0.0], [0.0, 10.0]])

    spec_labels = torch.tensor([0, 1])
    sev_labels = torch.tensor([0, 1])

    exporter.add_batch(
        ["s1", "s2"],
        ["train", "train"],
        ["src", "src"],
        ["en", "en"],
        spec_logits,
        sev_logits,
        spec_labels,
        sev_labels,
    )

    exporter.export()

    assert (tmp_path / "predictions.csv").exists()
    assert (tmp_path / "predictions.parquet").exists()
    assert (tmp_path / "misclassified.csv").exists()
    assert (tmp_path / "correct.csv").exists()
    assert (tmp_path / "confidence_distribution.csv").exists()
    assert (tmp_path / "entropy_distribution.csv").exists()

    df = pd.read_csv(tmp_path / "predictions.csv")
    assert len(df) == 2
    assert "entropy" in df.columns
    assert "department_entropy" in df.columns


def test_report_generation(tmp_path):
    config = MagicMock()
    config.__dict__ = {"optimizer": "adamw", "scheduler": "cosine"}
    generate_training_report(
        str(tmp_path), config, "exp_123", "abc1234", 120.5, 0.95, "hash_456"
    )

    assert (tmp_path / "training_summary.md").exists()
    assert (tmp_path / "training_metadata.json").exists()
    assert (tmp_path / "experiment_manifest.json").exists()
    assert (tmp_path / "hardware_report.json").exists()
    assert (tmp_path / "evaluation_summary.json").exists()

    with open(tmp_path / "training_metadata.json") as f:
        meta = json.load(f)
        assert meta["experiment_id"] == "exp_123"


def test_calibration(tmp_path):
    calibrator = Calibrator()

    # Needs some simulated poor calibration
    spec_logits = torch.tensor([[10.0, 0.0], [-10.0, 10.0]])  # very confident
    spec_labels = torch.tensor([1, 0])  # all wrong! -> high ECE

    sev_logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    sev_labels = torch.tensor([0, 1])  # correct

    report = calibrator.fit(
        spec_logits, spec_labels, sev_logits, sev_labels, str(tmp_path)
    )

    assert (tmp_path / "calibration_report.json").exists()
    assert "specialist" in report
    assert "severity" in report
    assert report["specialist"]["ece_before"] > 0
    assert "temperature" in report["specialist"]


def test_explainability_registration():
    model = MagicMock()
    registry = ExplainabilityRegistry(model)

    ig = registry.get_hook("ig")
    rollout = registry.get_hook("rollout")
    token = registry.get_hook("token")

    assert ig is not None
    assert rollout is not None
    assert token is not None

    ig.enable()
    assert ig.enabled == True

    res = ig.analyze(torch.tensor([[1, 2]]), torch.tensor([[1, 1]]), 0)
    assert res["method"] == "IntegratedGradients"


def test_artifact_integrity(tmp_path):
    # Dummy test to ensure all expected artifacts are written out with proper schema
    exporter = EvaluationExporter(str(tmp_path))
    exporter.add_batch(
        ["1"],
        ["test"],
        ["test"],
        ["en"],
        torch.tensor([[0.0, 1.0]]),
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([-1]),
        torch.tensor([-1]),  # masked labels
    )
    exporter.export()

    df = pd.read_csv(tmp_path / "predictions.csv")
    assert df.iloc[0]["ground_truth_department"] == "UNKNOWN"
