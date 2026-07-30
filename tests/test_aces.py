import torch

from models.emergent_path_triage.aces_utils import EvidenceDiagnostics
from models.emergent_path_triage.config import EmergentPathTriageConfig
from models.emergent_path_triage.dces import (
    AttentionFusion,
    ClinicalEvidenceSynthesizer,
    StaticFusion,
)


def test_static_fusion_fallback():
    """Test A0 mode (StaticFusion)."""
    config = EmergentPathTriageConfig(
        aces_fusion_mode="A0", latent_dim=16, dces_dropout=0.0
    )
    model = ClinicalEvidenceSynthesizer(hidden_dim=32, config=config)
    assert isinstance(model.fusion, StaticFusion)

    # Forward pass
    b, seq, hidden = 2, 10, 32
    token_emb = torch.randn(b, seq, hidden)
    mask = torch.ones(b, seq, dtype=torch.bool)

    out = model(token_emb, mask)
    assert out.symptom.shape == (b, 16)

    # Check trace
    trace = model.recorder.get_trace()
    assert trace is None or not model.recorder.record_enabled  # not enabled by default


def test_attention_fusion_a3_mode():
    """Test A3 mode (Prototypes + Residual)."""
    config = EmergentPathTriageConfig(
        aces_fusion_mode="A3", latent_dim=16, aces_num_heads=2
    )
    model = ClinicalEvidenceSynthesizer(hidden_dim=32, config=config)
    assert isinstance(model.fusion, AttentionFusion)

    # Enable recording
    model.recorder.record_enabled = True

    b, seq, hidden = 2, 10, 32
    token_emb = torch.randn(b, seq, hidden)
    mask = torch.ones(b, seq, dtype=torch.bool)

    out = model(token_emb, mask)
    assert out.symptom.shape == (b, 16)

    trace = model.recorder.get_trace()
    assert trace is not None
    assert trace.fusion_type == "AttentionFusion"

    # Diagnostics
    stats = EvidenceDiagnostics.aggregate_statistics(trace)
    assert "average_attention_entropy" in stats
    assert "per_class_aspect_importance" in stats
    assert "prototype_utilization" in stats

    # Gradients
    loss = (
        out.symptom.sum()
        + out.anatomical.sum()
        + out.temporal.sum()
        + out.systemic.sum()
    )
    loss.backward()

    # Check prototypes got gradients
    assert model.fusion.prototypes.grad is not None
    assert model.fusion.interaction.in_proj_weight.grad is not None


def test_attention_fusion_a1_mode():
    """Test A1 mode (Attention only, no prototypes)."""
    config = EmergentPathTriageConfig(
        aces_fusion_mode="A1", latent_dim=16, aces_num_heads=2
    )
    model = ClinicalEvidenceSynthesizer(hidden_dim=32, config=config)
    assert isinstance(model.fusion, AttentionFusion)
    assert not hasattr(model.fusion, "prototypes") or model.fusion.prototypes is None

    b, seq, hidden = 2, 10, 32
    token_emb = torch.randn(b, seq, hidden)
    mask = torch.ones(b, seq, dtype=torch.bool)
    out = model(token_emb, mask)
    assert out.symptom.shape == (b, 16)


def test_checkpoint_compatibility():
    """Test loading older checkpoints falls back to StaticFusion safely."""
    # Create model in A0 mode (older checkpoint would look like this, no fusion parameters)
    old_config = EmergentPathTriageConfig(aces_fusion_mode="A0")
    old_model = ClinicalEvidenceSynthesizer(hidden_dim=32, config=old_config)
    state_dict = old_model.state_dict()

    # Create new model in A3 mode
    new_config = EmergentPathTriageConfig(aces_fusion_mode="A3")
    new_model = ClinicalEvidenceSynthesizer(hidden_dim=32, config=new_config)

    # Load state dict without strict matching
    new_model.load_state_dict(state_dict, strict=False)

    # Should fall back to A0/StaticFusion
    assert isinstance(new_model.fusion, StaticFusion)
    assert new_model.config.aces_fusion_mode == "A0"
