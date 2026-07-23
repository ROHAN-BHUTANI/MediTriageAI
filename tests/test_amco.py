import pytest
import torch
import torch.nn as nn

from models.emergent_path_triage.config import EmergentPathTriageConfig
from models.emergent_path_triage.amco import StaticLossBalancer, HomoscedasticBalancer
from models.emergent_path_triage.amco_utils import OptimizationDiagnostics
from models.emergent_path_triage.model import EmergentPathTriageModel
from src.model import JointLoss


@pytest.fixture
def base_config():
    return EmergentPathTriageConfig(
        alpha_specialist=1.0,
        beta_severity=2.0,
        ortho_lambda=0.1,
        cons_lambda=0.2,
        div_lambda=0.3,
        amco_optimization_strategy="STATIC"
    )

@pytest.fixture
def homoscedastic_config():
    cfg = EmergentPathTriageConfig(
        amco_optimization_strategy="HOMOSCEDASTIC"
    )
    return cfg

def test_static_loss_balancer_equivalence(base_config):
    """Verify StaticLossBalancer is mathematically equivalent to the legacy fixed logic."""
    task_names = ["specialist", "severity", "ortho", "cons", "div"]
    balancer = StaticLossBalancer(base_config, task_names)
    
    losses = {
        "specialist": torch.tensor(1.5),
        "severity": torch.tensor(2.0),
        "ortho": torch.tensor(0.5),
        "cons": torch.tensor(0.1),
        "div": torch.tensor(0.8)
    }
    
    total_loss, effective_weights = balancer(losses)
    
    expected_total = (
        1.0 * 1.5 + 
        2.0 * 2.0 + 
        0.1 * 0.5 + 
        0.2 * 0.1 + 
        0.3 * 0.8
    )
    
    assert torch.isclose(total_loss, torch.tensor(expected_total))
    assert effective_weights["specialist"] == 1.0
    assert effective_weights["severity"] == 2.0

def test_homoscedastic_gradients(homoscedastic_config):
    """Verify Homoscedastic balancer parameters receive gradients correctly."""
    task_names = ["t1", "t2"]
    balancer = HomoscedasticBalancer(homoscedastic_config, task_names)
    
    # Enable recording for diagnostics
    balancer.recorder.record_enabled = True
    
    losses = {
        "t1": torch.tensor(2.0, requires_grad=True),
        "t2": torch.tensor(5.0, requires_grad=True)
    }
    
    total_loss, weights = balancer(losses)
    total_loss.backward()
    
    # Verify gradients flowed into the log_vars
    for task in task_names:
        assert balancer.log_vars[task].grad is not None
        
    # Verify diagnostic extraction
    trace = balancer.recorder.get_trace()
    assert trace is not None
    assert trace.optimization_type == "HOMOSCEDASTIC"
    assert "t1" in trace.effective_task_weights
    
    metrics = OptimizationDiagnostics.aggregate_statistics(trace)
    assert "optimization_entropy" in metrics
    assert "task_imbalance_index" in metrics
    assert "weight_t1" in metrics

def test_amco_checkpoint_compatibility():
    """Verify that a model loading older checkpoint falls back to StaticLossBalancer."""
    old_config = EmergentPathTriageConfig(amco_optimization_strategy="STATIC")
    old_model = EmergentPathTriageModel().build(old_config)
    state_dict = old_model.state_dict()
    
    # Remove any balancer keys just in case
    state_dict = {k: v for k, v in state_dict.items() if "loss_balancer" not in k}
    
    new_config = EmergentPathTriageConfig(amco_optimization_strategy="HOMOSCEDASTIC")
    new_model = EmergentPathTriageModel().build(new_config)
    
    assert isinstance(new_model.loss_balancer, HomoscedasticBalancer)
    
    # Load state dict without strict matching
    new_model.load_state_dict(state_dict, strict=False)
    
    # Should fall back to STATIC
    assert isinstance(new_model.loss_balancer, StaticLossBalancer)
    assert new_model.config.amco_optimization_strategy == "STATIC"

def test_model_compute_loss_integration(homoscedastic_config):
    """Verify model.compute_loss works correctly with AMCO."""
    model = EmergentPathTriageModel().build(homoscedastic_config)
    
    batch_size = 2
    spec_logits = torch.randn(batch_size, 13, requires_grad=True)
    sev_logits = torch.randn(batch_size, 5, requires_grad=True)
    
    labels_spec = torch.randint(0, 13, (batch_size,))
    labels_sev = torch.randint(0, 5, (batch_size,))
    
    joint_loss_fn = JointLoss()
    
    # Mock properties that compute_loss expects
    model._last_evidence = None
    model._last_final_state = None
    model._last_routing_decision = None
    homoscedastic_config.ortho_lambda = 0.0
    homoscedastic_config.cons_lambda = 0.0
    homoscedastic_config.div_lambda = 0.0
    
    loss_dict = model.compute_loss(
        spec_logits, sev_logits, labels_spec, labels_sev, joint_loss_fn
    )
    
    assert "joint_loss" in loss_dict
    assert "specialist_loss" in loss_dict
    assert "severity_loss" in loss_dict
    
    # The joint_loss should have gradients from spec_logits and sev_logits
    loss_dict["joint_loss"].backward()
    assert spec_logits.grad is not None
    assert sev_logits.grad is not None
