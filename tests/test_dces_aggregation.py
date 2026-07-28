import pytest
import torch
import torch.nn as nn
from models.emergent_path_triage.config import EmergentPathTriageConfig
from models.emergent_path_triage.model import EmergentPathTriageTransformer
from transformers import XLMRobertaConfig, XLMRobertaModel

@pytest.fixture
def mock_encoder():
    config = XLMRobertaConfig(
        vocab_size=100,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=256,
        max_position_embeddings=512,
    )
    return XLMRobertaModel(config)

@pytest.fixture
def config():
    return EmergentPathTriageConfig(
        latent_dim=64,
        routing_hidden_dim=64,
        num_thought_blocks=2,
    )

def test_dces_aggregation_initialization_equivalence(mock_encoder, config):
    model = EmergentPathTriageTransformer(mock_encoder, config)
    
    batch_size = 2
    latent_dim = config.latent_dim
    
    # Mock some evidence list
    z1 = torch.randn(batch_size, latent_dim)
    z2 = torch.randn(batch_size, latent_dim)
    z3 = torch.randn(batch_size, latent_dim)
    z4 = torch.randn(batch_size, latent_dim)
    
    evidence_list = [z1, z2, z3, z4]
    
    # Mean aggregation (previous approach)
    h_mean = torch.mean(torch.stack(evidence_list, dim=1), dim=1)
    
    # New projection aggregation
    fused = torch.cat(evidence_list, dim=-1)
    h_proj = model.evidence_projection(fused)
    
    # Check mathematical equivalence at initialization
    assert torch.allclose(h_mean, h_proj, atol=1e-6), "Projection is not equivalent to mean at init"

def test_dces_aggregation_backward_and_gradients(mock_encoder, config):
    model = EmergentPathTriageTransformer(mock_encoder, config)
    
    batch_size = 2
    latent_dim = config.latent_dim
    
    z1 = torch.randn(batch_size, latent_dim, requires_grad=True)
    z2 = torch.randn(batch_size, latent_dim, requires_grad=True)
    z3 = torch.randn(batch_size, latent_dim, requires_grad=True)
    z4 = torch.randn(batch_size, latent_dim, requires_grad=True)
    
    evidence_list = [z1, z2, z3, z4]
    fused = torch.cat(evidence_list, dim=-1)
    
    h_proj = model.evidence_projection(fused)
    loss = h_proj.sum()
    loss.backward()
    
    # Check gradients flow back to z1, z2, z3, z4
    assert z1.grad is not None and z1.grad.abs().sum() > 0
    assert z2.grad is not None and z2.grad.abs().sum() > 0
    
    # Check shape
    assert h_proj.shape == (batch_size, latent_dim)

def test_checkpoint_loading_behavior(mock_encoder, config, tmp_path):
    model1 = EmergentPathTriageTransformer(mock_encoder, config)
    
    # Save checkpoint
    ckpt_path = tmp_path / "model.pt"
    torch.save(model1.state_dict(), ckpt_path)
    
    # Create a new model and load checkpoint with strict=True
    # It should pass because we did NOT intercept load_state_dict and the checkpoint
    # contains evidence_projection if saved with this new model
    model2 = EmergentPathTriageTransformer(mock_encoder, config)
    model2.load_state_dict(torch.load(ckpt_path), strict=True)
    
    assert torch.allclose(model1.evidence_projection.weight, model2.evidence_projection.weight)
