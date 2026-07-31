import pytest
import torch

from models.emergent_path_triage.config import EmergentPathTriageConfig
from models.emergent_path_triage.dccf import (
    DirichletEstimator,
    IdentityEstimator,
    TemperatureScalingEstimator,
    VectorScalingEstimator,
)
from models.emergent_path_triage.dccf_utils import ClinicalConfidenceDiagnostics
from models.emergent_path_triage.model import EmergentPathTriageModel
from models.emergent_path_triage.types import ClinicalConfidenceTrace


@pytest.fixture
def config():
    return EmergentPathTriageConfig(
        num_thought_blocks=2, max_path_depth=2, latent_dim=128
    )


def test_identity_estimator(config):
    estimator = IdentityEstimator(config, 5)
    logits = torch.randn(4, 5)
    output = estimator.estimate(logits)

    # Uncalibrated softmax
    expected_probs = torch.softmax(logits, dim=-1)
    torch.testing.assert_close(output.calibrated_probabilities, expected_probs)

    expected_conf = expected_probs.max(dim=-1).values
    torch.testing.assert_close(output.confidence_score, expected_conf)

    # Metadata
    assert output.estimator_metadata["estimator"] == "IDENTITY"
    assert output.calibration_metadata["temperature"] == 1.0


def test_temperature_scaling_estimator(config):
    estimator = TemperatureScalingEstimator(config, 5)
    estimator.temperature.data = torch.tensor([2.0])

    logits = torch.randn(4, 5)
    output = estimator.estimate(logits)

    expected_probs = torch.softmax(logits / 2.0, dim=-1)
    torch.testing.assert_close(output.calibrated_probabilities, expected_probs)
    assert output.estimator_metadata["estimator"] == "TEMPERATURE"


def test_vector_scaling_estimator(config):
    estimator = VectorScalingEstimator(config, 5)
    estimator.W.data = torch.ones(5) * 0.5
    estimator.b.data = torch.ones(5) * 0.1

    logits = torch.randn(4, 5)
    output = estimator.estimate(logits)

    expected_probs = torch.softmax(logits * 0.5 + 0.1, dim=-1)
    torch.testing.assert_close(output.calibrated_probabilities, expected_probs)


def test_dirichlet_estimator(config):
    estimator = DirichletEstimator(config, 5)
    # Default is eye and zeros, so z = logits, alpha = softplus(logits) + 1e-10
    logits = torch.randn(4, 5)
    output = estimator.estimate(logits)

    z = logits
    alpha = torch.nn.functional.softplus(z) + 1e-10
    expected_probs = alpha / alpha.sum(dim=-1, keepdim=True)

    torch.testing.assert_close(output.calibrated_probabilities, expected_probs)


def test_confidence_telemetry_isolation(config):
    estimator = IdentityEstimator(config, 5)
    estimator.recorder.record_enabled = True
    logits = torch.randn(4, 5)

    estimator.estimate(logits)
    trace = estimator.recorder.get_trace()

    assert trace is not None
    assert isinstance(trace, ClinicalConfidenceTrace)
    torch.testing.assert_close(trace.raw_confidence, trace.calibrated_confidence)
    assert trace.entropy is not None


def test_diagnostics_metrics():
    # True labels
    labels = torch.tensor([0, 1, 2, 2])
    # Predicted probs
    probs = torch.tensor(
        [[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.3, 0.3, 0.4], [0.1, 0.2, 0.7]]
    )

    metrics = ClinicalConfidenceDiagnostics.compute_calibration_metrics(
        probs, labels, num_bins=10
    )

    assert "brier_score" in metrics
    assert "nll" in metrics
    assert "ece" in metrics
    assert "mce" in metrics
    assert metrics["brier_score"] >= 0
    assert metrics["nll"] >= 0


def test_model_integration_fallback(config):
    # Test strict=False fallback works cleanly with no DCCF keys
    model = EmergentPathTriageModel()
    net = model.build(config)

    # Save a raw dict WITHOUT DCCF keys
    state_dict = net.state_dict()
    keys_to_remove = [k for k in state_dict if "calibrator" in k]
    for k in keys_to_remove:
        del state_dict[k]

    config.dccf_confidence_estimator = "TEMPERATURE"
    net.load_state_dict(state_dict, strict=False)

    # Should revert to IDENTITY due to missing keys
    assert config.dccf_confidence_estimator == "IDENTITY"
    assert isinstance(net.specialist_calibrator, IdentityEstimator)


def test_hot_swapping(config):
    model = EmergentPathTriageModel()

    config.dccf_confidence_estimator = "VECTOR"
    net = model.build(config)
    assert isinstance(net.specialist_calibrator, VectorScalingEstimator)

    config.dccf_confidence_estimator = "DIRICHLET"
    net2 = model.build(config)
    assert isinstance(net2.specialist_calibrator, DirichletEstimator)
