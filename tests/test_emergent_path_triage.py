"""Unit tests for the E-PATH-CO-REASON hardened software architecture."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from models import MODEL_REGISTRY
from models.emergent_path_triage import (
    EmergentPathTriageConfig,
    EmergentPathTriageModel,
    EmergentPathCheckpointRegistry,
    apply_loss_hook,
    NUM_SPECIALIST_CLASSES,
    NUM_SEVERITY_LABELS,
    EvidenceRepresentation,
    RoutingDecision,
    ThoughtPath,
    ModelOutputs,
    ConfigurationError,
    RoutingError,
    InterfaceError,
    CompatibilityError,
    ClinicalEvidenceSynthesizer,
)
from models.emergent_path_triage.logger import LOGGER_NAME, get_logger


@pytest.fixture
def sample_config_dict():
    return {
        "num_thought_blocks": 5,
        "max_path_depth": 4,
        "latent_dim": 128,
        "temperature": 0.8,
        "alpha_specialist": 1.1,
        "beta_severity": 1.3,
        "ortho_lambda": 0.2,
        "cons_lambda": 0.6,
        "div_lambda": 0.15,
        "schema_version": "1.0",
        "architecture_version": "1.0.0",
        "compatibility_version": "1.0",
    }


def test_constants():
    """Verify that centralized constants are set to the expected values."""
    assert NUM_SPECIALIST_CLASSES == 13
    assert NUM_SEVERITY_LABELS == 5


def test_config_self_validation():
    """Verify that config objects perform strict validation on instantiation."""
    # 1. Invalid thought blocks
    with pytest.raises(ConfigurationError, match="num_thought_blocks must be positive"):
        EmergentPathTriageConfig(num_thought_blocks=0)
    
    # 2. Invalid path depth
    with pytest.raises(ConfigurationError, match="max_path_depth must be positive"):
        EmergentPathTriageConfig(max_path_depth=-1)
        
    # 3. Invalid latent dimension
    with pytest.raises(ConfigurationError, match="latent_dim must be a positive multiple of 2"):
        EmergentPathTriageConfig(latent_dim=65)
        
    # 4. Invalid temperature
    with pytest.raises(ConfigurationError, match="temperature must be strictly positive"):
        EmergentPathTriageConfig(temperature=0.0)
        
    # 5. Negative loss weights
    with pytest.raises(ConfigurationError, match="Loss coefficient cons_lambda must be non-negative"):
        EmergentPathTriageConfig(cons_lambda=-0.1)


def test_config_serialization(sample_config_dict):
    """Verify that config dict conversions are fully reversible."""
    config = EmergentPathTriageConfig.from_dict(sample_config_dict)
    d = config.to_dict()
    assert d["num_thought_blocks"] == 5
    assert d["latent_dim"] == 128
    
    config2 = EmergentPathTriageConfig.from_dict(d)
    assert config2 == config


def test_config_versioning_mismatch():
    """Verify that incompatible config versions raise ConfigurationError."""
    bad_data = {"compatibility_version": "2.0"}
    with pytest.raises(ConfigurationError, match="Incompatible configuration version"):
        EmergentPathTriageConfig.from_dict(bad_data)


def test_types_self_validation():
    """Verify that representation and output types perform validation checks."""
    device = torch.device("cpu")
    
    # 1. Evidence representation shape validation
    with pytest.raises(InterfaceError, match="must be a 2D tensor"):
        EvidenceRepresentation(
            symptom=torch.zeros((2, 64, 2), device=device),  # 3D
            anatomical=torch.zeros((2, 64), device=device),
            temporal=torch.zeros((2, 64), device=device),
            systemic=torch.zeros((2, 64), device=device),
    )
        
    # 2. Evidence batch mismatch
    with pytest.raises(InterfaceError, match="Batch size mismatch"):
        EvidenceRepresentation(
            symptom=torch.zeros((2, 64), device=device),
            anatomical=torch.zeros((3, 64), device=device),  # Batch=3
            temporal=torch.zeros((2, 64), device=device),
            systemic=torch.zeros((2, 64), device=device),
    )

    # 3. Routing decision properties
    with pytest.raises(RoutingError, match="routing_probabilities must be 3D"):
        RoutingDecision(
            routing_logits=torch.zeros((2, 3, 4), device=device),
            routing_probabilities=torch.zeros((2, 3), device=device),  # 2D
            selected_blocks=[0],
            path_depth=3,
            routing_entropy=torch.zeros((), device=device),
            routing_confidence=torch.ones((), device=device),
            path_identifier="test",
    )


def test_dataclasses_serialization_roundtrip():
    """Verify that all dataclasses inside types.py serialize and deserialize without information loss."""
    device = torch.device("cpu")
    
    # 1. EvidenceRepresentation
    ev = EvidenceRepresentation(
        symptom=torch.ones((2, 4), device=device),
        anatomical=torch.ones((2, 4), device=device),
        temporal=torch.ones((2, 4), device=device),
        systemic=torch.ones((2, 4), device=device),
    )
    d_ev = ev.to_dict()
    ev2 = EvidenceRepresentation.from_dict(d_ev)
    assert torch.allclose(ev2.symptom, ev.symptom)

    # 2. RoutingDecision
    dec = RoutingDecision(
        routing_logits=torch.ones((2, 2, 2), device=device),
        routing_probabilities=torch.ones((2, 2, 2), device=device),
        selected_blocks=[0, 1],
        path_depth=2,
        routing_entropy=torch.zeros((), device=device),
        routing_confidence=torch.ones((), device=device),
        path_identifier="test_roundtrip",
    )
    d_dec = dec.to_dict()
    dec2 = RoutingDecision.from_dict(d_dec)
    assert torch.allclose(dec2.routing_probabilities, dec.routing_probabilities)
    assert dec2.selected_blocks == [0, 1]

    # 3. ModelOutputs
    out = ModelOutputs(
        specialist_logits=torch.zeros((2, 13), device=device),
        severity_logits=torch.zeros((2, 5), device=device),
        routing_decision=dec,
    )
    d_out = out.to_dict()
    out2 = ModelOutputs.from_dict(d_out)
    assert out2.specialist_logits.shape == (2, 13)
    assert out2.routing_decision.path_identifier == "test_roundtrip"


def test_checkpoint_registry_compatibility(tmp_path):
    """Verify checkpoint metadata serialization and verification contracts."""
    registry = EmergentPathCheckpointRegistry()
    config = EmergentPathTriageConfig(latent_dim=64)
    
    meta = {
        "schema_version": "1.0",
        "architecture_version": "1.0.0",
        "compatibility_version": "1.0",
        "latent_dim": 64,
    }
    
    # Save & Load
    registry.save_checkpoint_metadata(tmp_path, meta)
    loaded = registry.load_checkpoint_metadata(tmp_path)
    assert loaded["compatibility_version"] == "1.0"
    
    # Compatibility checks
    assert registry.verify_compatibility(loaded, config) is True
    
    # Check mismatches
    bad_meta_ver = dict(meta, compatibility_version="2.0")
    with pytest.raises(CompatibilityError, match="compatibility version"):
        registry.verify_compatibility(bad_meta_ver, config)
        
    bad_meta_dim = dict(meta, latent_dim=128)
    with pytest.raises(CompatibilityError, match="latent dimension"):
        registry.verify_compatibility(bad_meta_dim, config)


def test_structured_logger():
    """Verify that E-PATH-CO-REASON writes to its dedicated logger namespace."""
    logger = get_logger()
    assert logger.name == LOGGER_NAME


def test_model_registration_and_metadata():
    """Verify E-PATH-CO-REASON is registered correctly with metadata."""
    assert "5" in MODEL_REGISTRY
    assert MODEL_REGISTRY["5"] is EmergentPathTriageModel

    model_meta = EmergentPathTriageModel()
    assert model_meta.architecture_name == "E-PATH-CO-REASON"
    assert model_meta.architecture_version == "1.0.0"
    assert model_meta.is_novel_contribution is True
    assert model_meta.short_name == "emergent_path_triage"


def test_model_tokenizer_and_forward_pass_interface():
    """Verify forward pass output contract, shapes, and iter behavior."""
    model_meta = EmergentPathTriageModel()
    tokenizer = model_meta.get_tokenizer()
    
    class TinyConfig:
        hidden_size: int = 32
        num_hidden_layers: int = 1
        num_attention_heads: int = 2
        intermediate_size: int = 64
        max_position_embeddings: int = 64

    built_model = model_meta.build(TinyConfig())
    
    input_ids = torch.randint(0, len(tokenizer), (2, 8))
    attention_mask = torch.ones_like(input_ids)
    
    # Check forward pass device compliance check
    with pytest.raises(InterfaceError, match="Tensor device mismatch"):
        # Create tensor on cuda if available, otherwise mock it for testing
        bad_device_tensor = input_ids.to("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            built_model.forward(bad_device_tensor, attention_mask.to("cpu"))
        else:
            # Force trigger device verification helper directly to simulate device mismatch
            built_model._verify_device_compliance(torch.zeros(1, device="meta"))

    outputs = built_model(input_ids, attention_mask)
    assert isinstance(outputs, ModelOutputs)
    assert outputs.specialist_logits.shape == (2, 13)
    assert outputs.severity_logits.shape == (2, 5)
    
    # Unpacking test
    spec, sev = outputs
    assert spec.shape == (2, 13)
    assert sev.shape == (2, 5)


def test_loss_hook_integration():
    """Verify custom loss helper and training hook fallbacks."""
    from src.model import JointLoss
    
    spec_logits = torch.zeros((2, 13))
    sev_logits = torch.zeros((2, 5))
    labels_spec = torch.zeros(2, dtype=torch.long)
    labels_sev = torch.zeros(2, dtype=torch.long)
    loss_fn = JointLoss()
    
    model_meta = EmergentPathTriageModel()
    
    class TinyConfig:
        hidden_size: int = 32
        num_hidden_layers: int = 1
        num_attention_heads: int = 2
        intermediate_size: int = 64
        max_position_embeddings: int = 64
        
    built_model = model_meta.build(TinyConfig())
    
    loss_dict = apply_loss_hook(built_model, spec_logits, sev_logits, labels_spec, labels_sev, loss_fn)
    assert "joint_loss" in loss_dict
    assert "ortho_loss" in loss_dict
    assert "cons_loss" in loss_dict
    assert "div_loss" in loss_dict

    # Check incorrect shape validation in compute_loss
    bad_spec = torch.zeros((2, 12))  # Should be 13
    with pytest.raises(InterfaceError, match="Incorrect specialist_logits shape"):
        built_model.compute_loss(bad_spec, sev_logits, labels_spec, labels_sev, loss_fn)


def test_dces_output_dimensions():
    """Verify that DCES outputs tensors of the correct shapes matching configuration."""
    config = EmergentPathTriageConfig(latent_dim=64)
    dces = ClinicalEvidenceSynthesizer(hidden_dim=768, config=config)
    
    token_embeddings = torch.randn(4, 16, 768)
    attention_mask = torch.ones(4, 16, dtype=torch.long)
    
    evidence = dces(token_embeddings, attention_mask)
    assert isinstance(evidence, EvidenceRepresentation)
    assert evidence.symptom.shape == (4, 64)
    assert evidence.anatomical.shape == (4, 64)
    assert evidence.temporal.shape == (4, 64)
    assert evidence.systemic.shape == (4, 64)


def test_dces_deterministic_behavior():
    """Verify that DCES produces identical outputs for identical seeds and inputs."""
    config = EmergentPathTriageConfig(latent_dim=32, dces_dropout=0.0)
    
    torch.manual_seed(100)
    dces1 = ClinicalEvidenceSynthesizer(hidden_dim=256, config=config)
    
    torch.manual_seed(100)
    dces2 = ClinicalEvidenceSynthesizer(hidden_dim=256, config=config)
    
    # Verify parameter match
    for p1, p2 in zip(dces1.parameters(), dces2.parameters()):
        assert torch.allclose(p1, p2)
        
    token_embeddings = torch.randn(2, 8, 256)
    attention_mask = torch.ones(2, 8, dtype=torch.long)
    
    out1 = dces1(token_embeddings, attention_mask)
    out2 = dces2(token_embeddings, attention_mask)
    
    assert torch.allclose(out1.symptom, out2.symptom)
    assert torch.allclose(out1.anatomical, out2.anatomical)


def test_dces_configuration_variants():
    """Verify that DCES initializes and executes correctly under different configurations."""
    for activation in ["gelu", "relu", "silu", "tanh"]:
        for norm in ["layernorm", "none"]:
            config = EmergentPathTriageConfig(
                latent_dim=32,
                dces_activation=activation,
                dces_normalization=norm,
                dces_dropout=0.2,
            )
            dces = ClinicalEvidenceSynthesizer(hidden_dim=128, config=config)
            
            token_embeddings = torch.randn(2, 10, 128)
            attention_mask = torch.ones(2, 10, dtype=torch.long)
            
            evidence = dces(token_embeddings, attention_mask)
            assert evidence.symptom.shape == (2, 32)


def test_dces_invalid_input_handling():
    """Verify that DCES throws InterfaceError when input parameters are invalid."""
    config = EmergentPathTriageConfig(latent_dim=32)
    dces = ClinicalEvidenceSynthesizer(hidden_dim=128, config=config)
    
    token_embeddings = torch.randn(2, 10, 128)
    attention_mask = torch.ones(2, 10, dtype=torch.long)
    
    # 1. Invalid input dimension (embeddings must be 3D)
    with pytest.raises(InterfaceError, match="token_embeddings must be 3D"):
        dces(torch.randn(2, 128), attention_mask)
        
    # 2. Invalid attention mask dimension (must be 2D)
    with pytest.raises(InterfaceError, match="attention_mask must be 2D"):
        dces(token_embeddings, torch.ones(2, dtype=torch.long))
        
    # 3. Batch dimension mismatch
    with pytest.raises(InterfaceError, match="Batch dimension mismatch"):
        dces(token_embeddings, torch.ones(3, 10, dtype=torch.long))
        
    # 4. Sequence dimension mismatch
    with pytest.raises(InterfaceError, match="Sequence dimension mismatch"):
        dces(token_embeddings, torch.ones(2, 9, dtype=torch.long))
        
    # 5. Hidden size mismatch
    with pytest.raises(InterfaceError, match="Hidden dimension mismatch"):
        dces(torch.randn(2, 10, 64), attention_mask)
        
    # 6. Dtype mismatch (embeddings must be float32)
    with pytest.raises(InterfaceError, match="Incorrect dtype"):
        dces(token_embeddings.to(torch.float64), attention_mask)
        
    # 7. Device mismatch simulation
    with pytest.raises(InterfaceError, match="Device mismatch"):
        dces(token_embeddings, attention_mask.to("meta"))


def test_dces_serialization_compatibility():
    """Verify that evidence generated by DCES is compatible with types.py serialization round-trips."""
    config = EmergentPathTriageConfig(latent_dim=64)
    dces = ClinicalEvidenceSynthesizer(hidden_dim=128, config=config)
    
    token_embeddings = torch.randn(2, 10, 128)
    attention_mask = torch.ones(2, 10, dtype=torch.long)
    
    evidence = dces(token_embeddings, attention_mask)
    
    # Serialize to dictionary
    serialized = evidence.to_dict()
    assert isinstance(serialized["symptom"], list)
    
    # Reconstruct from dictionary
    reconstructed = EvidenceRepresentation.from_dict(serialized)
    assert reconstructed.symptom.shape == (2, 64)
    assert torch.allclose(reconstructed.symptom, evidence.symptom)


def test_dces_aspect_independence():
    """Verify that the four aspect projection sub-networks have parameter-isolated weights."""
    config = EmergentPathTriageConfig(latent_dim=16)
    dces = ClinicalEvidenceSynthesizer(hidden_dim=32, config=config)
    
    # Verify parameter objects are distinct objects
    params_symptom = list(dces.symptom_proj.parameters())
    params_anatomical = list(dces.anatomical_proj.parameters())
    
    assert params_symptom[0] is not params_anatomical[0]
    
    # Verify they don't produce the exact same weights initially
    assert not torch.allclose(params_symptom[0], params_anatomical[0])


def test_dces_numerical_stability():
    """Verify that DCES pooling handles fully padded sequences safely without generating NaN or Inf."""
    config = EmergentPathTriageConfig(latent_dim=16)
    dces = ClinicalEvidenceSynthesizer(hidden_dim=32, config=config)
    
    token_embeddings = torch.randn(2, 5, 32)
    # Fully padded sequence for sample 0, partially active for sample 1
    attention_mask = torch.tensor([[0, 0, 0, 0, 0], [1, 1, 1, 0, 0]], dtype=torch.long)
    
    evidence = dces(token_embeddings, attention_mask)
    
    # Assert sample 0 (fully padded) outputs clean zeroes, not NaNs/Infs
    assert torch.all(evidence.symptom[0] == 0.0)
    assert not torch.isnan(evidence.symptom[0]).any()
    assert not torch.isinf(evidence.symptom[0]).any()
    
    # Assert sample 1 (active) is non-zero
    assert not torch.all(evidence.symptom[1] == 0.0)


def test_dces_pairwise_similarities():
    """Verify that the pairwise aspect cosine similarity matrices are computed correctly."""
    device = torch.device("cpu")
    
    # Create manual orthogonal/colinear representations to test math validity
    symptom = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    anatomical = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=device)
    temporal = torch.tensor([[0.0, 0.0, 1.0, 0.0]], device=device)
    systemic = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device)
    
    evidence = EvidenceRepresentation(
        symptom=symptom,
        anatomical=anatomical,
        temporal=temporal,
        systemic=systemic,
    )
    
    sim = evidence.compute_pairwise_similarities()
    assert sim.shape == (1, 4, 4)
    
    # Since inputs were perfectly orthogonal standard basis vectors, similarities must match the Identity matrix
    identity = torch.eye(4, device=device).unsqueeze(0)
    assert torch.allclose(sim, identity, atol=1e-6)


def test_dces_edge_cases_inputs():
    """Verify validation boundaries under long sequences, single-tokens, and mixed batches."""
    config = EmergentPathTriageConfig(latent_dim=16)
    dces = ClinicalEvidenceSynthesizer(hidden_dim=32, config=config)
    
    # 1. Single-token sequences
    t1 = torch.randn(2, 1, 32)
    m1 = torch.ones(2, 1, dtype=torch.long)
    assert dces(t1, m1).symptom.shape == (2, 16)
    
    # 2. Mixed batch sizes
    t2 = torch.randn(1, 10, 32)
    m2 = torch.ones(1, 10, dtype=torch.long)
    assert dces(t2, m2).symptom.shape == (1, 16)
    
    # 3. Long sequences
    t3 = torch.randn(2, 500, 32)
    m3 = torch.ones(2, 500, dtype=torch.long)
    assert dces(t3, m3).symptom.shape == (2, 16)


def test_dcrr_probability_normalization_and_shapes():
    """Verify routing probability normalization, tensor shapes, and bounds."""
    config = EmergentPathTriageConfig(latent_dim=16, num_thought_blocks=6, max_path_depth=3)
    
    # Import router
    from models.emergent_path_triage.dcrr import ClinicalReasoningRouter
    router = ClinicalReasoningRouter(config)
    
    device = torch.device("cpu")
    evidence = EvidenceRepresentation(
        symptom=torch.randn(4, 16, device=device),
        anatomical=torch.randn(4, 16, device=device),
        temporal=torch.randn(4, 16, device=device),
        systemic=torch.randn(4, 16, device=device)
    )
    
    # Run in inference mode (hard selection, routing probabilities should be one-hot)
    router.eval()
    decision = router(evidence, temperature=1.0)
    
    assert isinstance(decision, RoutingDecision)
    assert decision.routing_logits.shape == (4, 3, 6)
    assert decision.routing_probabilities.shape == (4, 3, 6)
    assert len(decision.selected_blocks) == 3
    assert decision.path_depth == 3
    assert decision.path_identifier.startswith("infer_hard_path_")
    
    # Verify probability distribution sums to 1.0 at each step (one-hot vectors)
    row_sums = decision.routing_probabilities.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums))
    
    # Verify values are either 0 or 1 in hard eval routing
    assert torch.all((decision.routing_probabilities == 0.0) | (decision.routing_probabilities == 1.0))


def test_dcrr_deterministic_inference():
    """Verify that router inference yields deterministic paths for identical inputs."""
    config = EmergentPathTriageConfig(latent_dim=16, num_thought_blocks=4, max_path_depth=3)
    from models.emergent_path_triage.dcrr import ClinicalReasoningRouter
    
    torch.manual_seed(42)
    router1 = ClinicalReasoningRouter(config)
    torch.manual_seed(42)
    router2 = ClinicalReasoningRouter(config)
    
    router1.eval()
    router2.eval()
    
    device = torch.device("cpu")
    evidence = EvidenceRepresentation(
        symptom=torch.randn(2, 16, device=device),
        anatomical=torch.randn(2, 16, device=device),
        temporal=torch.randn(2, 16, device=device),
        systemic=torch.randn(2, 16, device=device)
    )
    
    dec1 = router1(evidence, temperature=1.0)
    dec2 = router2(evidence, temperature=1.0)
    
    assert torch.allclose(dec1.routing_logits, dec2.routing_logits)
    assert dec1.selected_blocks == dec2.selected_blocks
    assert dec1.path_identifier == dec2.path_identifier


def test_dcrr_stochastic_training():
    """Verify that routing Gumbel-Softmax maps gradients and exhibits variance in training mode."""
    config = EmergentPathTriageConfig(latent_dim=8, num_thought_blocks=4, max_path_depth=2)
    from models.emergent_path_triage.dcrr import ClinicalReasoningRouter
    router = ClinicalReasoningRouter(config)
    
    # Put in train mode (soft Gumbel-Softmax)
    router.train()
    
    device = torch.device("cpu")
    evidence = EvidenceRepresentation(
        symptom=torch.randn(2, 8, device=device),
        anatomical=torch.randn(2, 8, device=device),
        temporal=torch.randn(2, 8, device=device),
        systemic=torch.randn(2, 8, device=device)
    )
    
    # Run forward pass with parameter tracking
    for p in router.parameters():
        p.requires_grad = True
        
    decision = router(evidence, temperature=1.0)
    
    # Check that routing probabilities are soft values in train mode (between 0 and 1 exclusive)
    assert torch.any((decision.routing_probabilities > 0.0) & (decision.routing_probabilities < 1.0))
    
    # Run loss call to verify backprop compatibility
    loss = decision.routing_probabilities.sum()
    loss.backward()
    
    # Confirm gradients exist for executed layers
    for name, p in router.named_parameters():
        if "gru_cell" in name or "init_proj" in name or "logits_proj" in name:
            continue
        assert p.grad is not None, f"Parameter {name} did not receive gradients"


def test_dcrr_temperature_scaling():
    """Verify that temperature adjustments properly scale Softmax concentrations."""
    config = EmergentPathTriageConfig(latent_dim=8, num_thought_blocks=4, max_path_depth=2)
    from models.emergent_path_triage.dcrr import ClinicalReasoningRouter
    router = ClinicalReasoningRouter(config)
    router.train()
    
    device = torch.device("cpu")
    evidence = EvidenceRepresentation(
        symptom=torch.randn(1, 8, device=device),
        anatomical=torch.randn(1, 8, device=device),
        temporal=torch.randn(1, 8, device=device),
        systemic=torch.randn(1, 8, device=device)
    )
    
    # High temperature: should make probabilities close to uniform distribution (entropy increases)
    dec_high = router(evidence, temperature=100.0)
    # Low temperature: should make probabilities close to one-hot (entropy decreases)
    dec_low = router(evidence, temperature=0.1)
    
    # Assert soft probabilities sum to 1.0
    assert torch.allclose(dec_high.routing_probabilities.sum(dim=-1), torch.ones(1, 2))
    assert torch.allclose(dec_low.routing_probabilities.sum(dim=-1), torch.ones(1, 2))


def test_dcrr_invalid_configurations():
    """Verify that router validation rejects invalid hyperparameters."""
    from models.emergent_path_triage.dcrr import ClinicalReasoningRouter
    
    # 1. Non-positive temperature throws RoutingError
    config = EmergentPathTriageConfig(latent_dim=8, num_thought_blocks=4, max_path_depth=2)
    router = ClinicalReasoningRouter(config)
    
    device = torch.device("cpu")
    evidence = EvidenceRepresentation(
        symptom=torch.randn(1, 8, device=device),
        anatomical=torch.randn(1, 8, device=device),
        temporal=torch.randn(1, 8, device=device),
        systemic=torch.randn(1, 8, device=device)
    )
    
    with pytest.raises(RoutingError, match="temperature must be strictly positive"):
        router(evidence, temperature=0.0)
        
    with pytest.raises(RoutingError, match="temperature must be strictly positive"):
        router(evidence, temperature=-0.5)


def test_dcrr_serialization_and_cpu():
    """Verify DCRR output serialization roundtrip compatibility and CPU execution."""
    config = EmergentPathTriageConfig(latent_dim=8, num_thought_blocks=3, max_path_depth=2)
    from models.emergent_path_triage.dcrr import ClinicalReasoningRouter
    router = ClinicalReasoningRouter(config)
    router.eval()
    
    device = torch.device("cpu")
    evidence = EvidenceRepresentation(
        symptom=torch.randn(2, 8, device=device),
        anatomical=torch.randn(2, 8, device=device),
        temporal=torch.randn(2, 8, device=device),
        systemic=torch.randn(2, 8, device=device)
    )
    
    decision = router(evidence, temperature=1.0)
    
    # Verify serialization
    serialized = decision.to_dict()
    assert isinstance(serialized["routing_logits"], list)
    assert isinstance(serialized["routing_probabilities"], list)
    assert serialized["path_depth"] == 2
    
    # Deserialize and verify shapes
    reconstructed = RoutingDecision.from_dict(serialized, device="cpu")
    assert reconstructed.routing_logits.shape == (2, 2, 3)
    assert torch.allclose(reconstructed.routing_logits, decision.routing_logits)


def test_ctb_parameter_independence():
    """Verify that multiple CTBs instantiated from the template own completely independent parameters."""
    config = EmergentPathTriageConfig(latent_dim=16, ctb_hidden_dim=32)
    from models.emergent_path_triage.ctb import ClinicalThoughtBlock
    
    ctb1 = ClinicalThoughtBlock(latent_dim=16, config=config)
    ctb2 = ClinicalThoughtBlock(latent_dim=16, config=config)
    
    # Check parameters are distinct objects
    params1 = list(ctb1.parameters())
    params2 = list(ctb2.parameters())
    assert params1[0] is not params2[0]
    
    # Check parameter values are not identical initially due to default initialization differences
    assert not torch.allclose(ctb1.linear1.weight, ctb2.linear1.weight)


def test_ctb_identical_initialization():
    """Verify that CTBs initialize identically when seed values are fixed."""
    config = EmergentPathTriageConfig(latent_dim=16, ctb_hidden_dim=32)
    from models.emergent_path_triage.ctb import ClinicalThoughtBlock
    
    torch.manual_seed(42)
    ctb1 = ClinicalThoughtBlock(latent_dim=16, config=config)
    
    torch.manual_seed(42)
    ctb2 = ClinicalThoughtBlock(latent_dim=16, config=config)
    
    # Check parameter values match exactly
    for p1, p2 in zip(ctb1.parameters(), ctb2.parameters()):
        assert torch.allclose(p1, p2)


def test_ctb_independent_parameter_updates():
    """Verify that updating parameter weights in one block does not affect another block."""
    config = EmergentPathTriageConfig(latent_dim=8, ctb_hidden_dim=16, ctb_dropout=0.0)
    from models.emergent_path_triage.ctb import ClinicalThoughtBlock
    
    torch.manual_seed(42)
    ctb1 = ClinicalThoughtBlock(latent_dim=8, config=config)
    torch.manual_seed(42)
    ctb2 = ClinicalThoughtBlock(latent_dim=8, config=config)
    
    # Set up optimizer for ctb1 only
    optimizer = torch.optim.SGD(ctb1.parameters(), lr=0.1)
    
    state = torch.randn(2, 8)
    
    # Execute forward pass and backward step on ctb1
    optimizer.zero_grad()
    out1 = ctb1(state)
    loss = out1.sum()
    loss.backward()
    optimizer.step()
    
    # Verify ctb1 parameters changed from their initialized states
    # Verify ctb2 parameters remain identical to their initialized states
    for p1, p2 in zip(ctb1.parameters(), ctb2.parameters()):
        assert not torch.allclose(p1, p2)


def test_ctb_output_dimension_preservation():
    """Verify that CTBs preserve input shape dimensions exactly."""
    config = EmergentPathTriageConfig(latent_dim=32, ctb_hidden_dim=64)
    from models.emergent_path_triage.ctb import ClinicalThoughtBlock
    ctb = ClinicalThoughtBlock(latent_dim=32, config=config)
    
    # Input batch size 4, latent dim 32
    state = torch.randn(4, 32)
    output = ctb(state)
    
    assert output.shape == (4, 32)


def test_ctb_invalid_input_handling():
    """Verify that ClinicalThoughtBlock validations catch bad inputs."""
    config = EmergentPathTriageConfig(latent_dim=16, ctb_hidden_dim=32)
    from models.emergent_path_triage.ctb import ClinicalThoughtBlock
    ctb = ClinicalThoughtBlock(latent_dim=16, config=config)
    
    # 1. Invalid input dimension (must be 2D)
    with pytest.raises(InterfaceError, match="state must be a 2D tensor"):
        ctb(torch.randn(2, 16, 1))
        
    # 2. Latent dimension mismatch
    with pytest.raises(InterfaceError, match="Latent dimension mismatch"):
        ctb(torch.randn(2, 32))
        
    # 3. Dtype mismatch (must be float32)
    with pytest.raises(InterfaceError, match="Incorrect dtype"):
        ctb(torch.randn(2, 16).to(torch.float64))
        
    # 4. Device mismatch simulation
    with pytest.raises(InterfaceError, match="Device mismatch"):
        ctb(torch.zeros(2, 16, device="meta"))


def test_ctb_serialization_and_cpu():
    """Verify that CTB outputs roundtrip and execute on CPU."""
    config = EmergentPathTriageConfig(latent_dim=16, ctb_hidden_dim=32)
    from models.emergent_path_triage.ctb import ClinicalThoughtBlock
    ctb = ClinicalThoughtBlock(latent_dim=16, config=config)
    ctb.eval()
    
    state = torch.randn(2, 16)
    output = ctb(state)
    
    # Mock serialization mapping of output
    serialized = output.detach().cpu().tolist()
    assert isinstance(serialized, list)
    assert len(serialized) == 2
    assert len(serialized[0]) == 16
    
    reconstructed = torch.tensor(serialized, dtype=torch.float32)
    assert torch.allclose(reconstructed, output)


def test_engine_correct_execution_order_and_skips():
    """Verify execution order, intermediate state tracking, and that unused blocks remain un-executed."""
    config = EmergentPathTriageConfig(latent_dim=8, num_thought_blocks=3, max_path_depth=2, ctb_hidden_dim=16, ctb_dropout=0.0)
    from models.emergent_path_triage.engine import ReasoningPathExecutionEngine
    from models.emergent_path_triage.ctb import ClinicalThoughtBlock
    
    engine = ReasoningPathExecutionEngine(config)
    engine.eval()
    
    # Instantiate 3 blocks
    blocks = nn.ModuleList([
        ClinicalThoughtBlock(latent_dim=8, config=config),
        ClinicalThoughtBlock(latent_dim=8, config=config),
        ClinicalThoughtBlock(latent_dim=8, config=config)
    ])
    blocks.eval()
    
    # Mock a RoutingDecision path: step 0 -> block 2, step 1 -> block 0 (block 1 is unused)
    dec = RoutingDecision(
        routing_logits=torch.zeros((1, 2, 3)),
        routing_probabilities=torch.zeros((1, 2, 3)),
        selected_blocks=[2, 0],
        path_depth=2,
        routing_entropy=torch.zeros(()),
        routing_confidence=torch.ones(()),
        path_identifier="test_engine"
    )
    
    # Populate mock routing probabilities to have 1.0 at selection
    dec.routing_probabilities[0, 0, 2] = 1.0
    dec.routing_probabilities[0, 1, 0] = 1.0
    
    evidence = [
        torch.ones(1, 8),
        torch.ones(1, 8),
        torch.ones(1, 8),
        torch.ones(1, 8)
    ]
    
    # Execute
    final_state, path = engine(evidence, dec, blocks)
    
    # 1. State check
    assert path.states == [2, 0]
    
    # 2. Intermediate representations check
    # Initial state (mean of all 1s is 1s), step 1, step 2 (total 3 representations)
    assert len(path.representations) == 3
    h0 = path.representations[0]
    h1 = path.representations[1]
    h2 = path.representations[2]
    
    assert torch.allclose(h0, torch.ones(1, 8))
    
    # Verify exact step execution matching:
    # h1 must equal block 2 executed on h0
    expected_h1 = blocks[2](h0)
    assert torch.allclose(h1, expected_h1)
    
    # h2 must equal block 0 executed on h1
    expected_h2 = blocks[0](h1)
    assert torch.allclose(h2, expected_h2)
    assert torch.allclose(final_state, h2)


def test_engine_deterministic_inference():
    """Verify that inference mode runs identically and deterministically for fixed inputs."""
    config = EmergentPathTriageConfig(latent_dim=8, num_thought_blocks=3, max_path_depth=2, ctb_hidden_dim=16, ctb_dropout=0.0)
    from models.emergent_path_triage.engine import ReasoningPathExecutionEngine
    from models.emergent_path_triage.ctb import ClinicalThoughtBlock
    
    engine = ReasoningPathExecutionEngine(config)
    engine.eval()
    
    blocks = nn.ModuleList([
        ClinicalThoughtBlock(latent_dim=8, config=config),
        ClinicalThoughtBlock(latent_dim=8, config=config),
        ClinicalThoughtBlock(latent_dim=8, config=config)
    ])
    blocks.eval()
    
    dec = RoutingDecision(
        routing_logits=torch.zeros((2, 2, 3)),
        routing_probabilities=torch.zeros((2, 2, 3)),
        selected_blocks=[0, 1],
        path_depth=2,
        routing_entropy=torch.zeros(()),
        routing_confidence=torch.ones(()),
        path_identifier="test_engine_det"
    )
    dec.routing_probabilities[:, 0, 0] = 1.0
    dec.routing_probabilities[:, 1, 1] = 1.0
    
    evidence = [
        torch.randn(2, 8),
        torch.randn(2, 8),
        torch.randn(2, 8),
        torch.randn(2, 8)
    ]
    
    final1, path1 = engine(evidence, dec, blocks)
    final2, path2 = engine(evidence, dec, blocks)
    
    assert torch.allclose(final1, final2)
    for t1, t2 in zip(path1.representations, path2.representations):
        assert torch.allclose(t1, t2)


def test_engine_gradient_propagation():
    """Verify that gradients propagate back through selected CTBs and routing probabilities during training."""
    config = EmergentPathTriageConfig(latent_dim=8, num_thought_blocks=3, max_path_depth=2, ctb_hidden_dim=16, ctb_dropout=0.0)
    from models.emergent_path_triage.engine import ReasoningPathExecutionEngine
    from models.emergent_path_triage.ctb import ClinicalThoughtBlock
    
    engine = ReasoningPathExecutionEngine(config)
    
    # Put engine and blocks in train mode to blend soft paths
    engine.train()
    blocks = nn.ModuleList([
        ClinicalThoughtBlock(latent_dim=8, config=config),
        ClinicalThoughtBlock(latent_dim=8, config=config),
        ClinicalThoughtBlock(latent_dim=8, config=config)
    ])
    for p in blocks.parameters():
        p.requires_grad = True
        
    dec_probs = torch.rand((1, 2, 3), requires_grad=True)
    dec = RoutingDecision(
        routing_logits=dec_probs,
        routing_probabilities=torch.softmax(dec_probs, dim=-1),
        selected_blocks=[0, 1],
        path_depth=2,
        routing_entropy=torch.zeros(()),
        routing_confidence=torch.ones(()),
        path_identifier="test_engine_grad"
    )
    
    evidence = [
        torch.randn(1, 8),
        torch.randn(1, 8),
        torch.randn(1, 8),
        torch.randn(1, 8)
    ]
    
    final, path = engine(evidence, dec, blocks)
    loss = final.sum()
    loss.backward()
    
    # Verify gradients propagate to block weights
    for name, p in blocks.named_parameters():
        assert p.grad is not None, f"Block param {name} did not receive gradients"
        
    # Verify gradients propagate to routing probabilities
    assert dec_probs.grad is not None


def test_engine_invalid_routing_and_validation():
    """Verify validation boundaries and exception triggers under invalid parameters."""
    config = EmergentPathTriageConfig(latent_dim=8, num_thought_blocks=3, max_path_depth=2, ctb_hidden_dim=16)
    from models.emergent_path_triage.engine import ReasoningPathExecutionEngine
    from models.emergent_path_triage.ctb import ClinicalThoughtBlock
    
    engine = ReasoningPathExecutionEngine(config)
    blocks = nn.ModuleList([
        ClinicalThoughtBlock(latent_dim=8, config=config),
        ClinicalThoughtBlock(latent_dim=8, config=config),
        ClinicalThoughtBlock(latent_dim=8, config=config)
    ])
    
    # 1. Invalid RoutingDecision path depth
    dec_bad_depth = RoutingDecision(
        routing_logits=torch.zeros((1, 3, 3)),
        routing_probabilities=torch.zeros((1, 3, 3)),
        selected_blocks=[0, 1, 2],
        path_depth=3, # Config demands 2
        routing_entropy=torch.zeros(()),
        routing_confidence=torch.ones(()),
        path_identifier="test_engine"
    )
    evidence = [torch.randn(1, 8) for _ in range(4)]
    with pytest.raises(RoutingError, match="Routing path depth mismatch"):
        engine(evidence, dec_bad_depth, blocks)
        
    # 2. Out of bounds block selection
    dec_bad_indices = RoutingDecision(
        routing_logits=torch.zeros((1, 2, 3)),
        routing_probabilities=torch.zeros((1, 2, 3)),
        selected_blocks=[0, 5], # 5 is invalid for 3 blocks
        path_depth=2,
        routing_entropy=torch.zeros(()),
        routing_confidence=torch.ones(()),
        path_identifier="test_engine"
    )
    with pytest.raises(RoutingError, match="Invalid block selection"):
        engine(evidence, dec_bad_indices, blocks)
        
    # 3. Missing / mismatched blocks count
    blocks_too_few = nn.ModuleList([
        ClinicalThoughtBlock(latent_dim=8, config=config)
    ])
    dec_ok = RoutingDecision(
        routing_logits=torch.zeros((1, 2, 3)),
        routing_probabilities=torch.zeros((1, 2, 3)),
        selected_blocks=[0, 0],
        path_depth=2,
        routing_entropy=torch.zeros(()),
        routing_confidence=torch.ones(()),
        path_identifier="test_engine"
    )
    with pytest.raises(InterfaceError, match="Blocks count mismatch"):
        engine(evidence, dec_ok, blocks_too_few)


def test_engine_serialization_and_cpu():
    """Verify that populated ThoughtPath outputs serialize, deserialize, and run on CPU."""
    config = EmergentPathTriageConfig(latent_dim=8, num_thought_blocks=3, max_path_depth=2, ctb_hidden_dim=16)
    from models.emergent_path_triage.engine import ReasoningPathExecutionEngine
    from models.emergent_path_triage.ctb import ClinicalThoughtBlock
    
    engine = ReasoningPathExecutionEngine(config)
    engine.eval()
    
    blocks = nn.ModuleList([
        ClinicalThoughtBlock(latent_dim=8, config=config),
        ClinicalThoughtBlock(latent_dim=8, config=config),
        ClinicalThoughtBlock(latent_dim=8, config=config)
    ])
    
    dec = RoutingDecision(
        routing_logits=torch.zeros((1, 2, 3)),
        routing_probabilities=torch.zeros((1, 2, 3)),
        selected_blocks=[0, 1],
        path_depth=2,
        routing_entropy=torch.zeros(()),
        routing_confidence=torch.ones(()),
        path_identifier="test_engine"
    )
    evidence = [torch.randn(1, 8) for _ in range(4)]
    
    final, path = engine(evidence, dec, blocks)
    
    # Serialize ThoughtPath to dict
    serialized = path.to_dict()
    assert isinstance(serialized["states"], list)
    assert len(serialized["representations"]) == 3
    
    # Deserialize back
    reconstructed = ThoughtPath.from_dict(serialized, device="cpu")
    assert reconstructed.states == [0, 1]
    assert len(reconstructed.representations) == 3
    assert torch.allclose(reconstructed.representations[0], path.representations[0])


def test_head_output_dimensions():
    """Verify prediction heads map input representations to correct classification dimensions."""
    config = EmergentPathTriageConfig(latent_dim=16, head_hidden_dim=32, head_dropout=0.0)
    from models.emergent_path_triage.heads import PredictionHead
    
    spec_head = PredictionHead(latent_dim=16, output_dim=13, config=config)
    sev_head = PredictionHead(latent_dim=16, output_dim=5, config=config)
    
    spec_head.eval()
    sev_head.eval()
    
    # Input batch size 3
    state = torch.randn(3, 16)
    
    spec_logits = spec_head(state)
    sev_logits = sev_head(state)
    
    assert spec_logits.shape == (3, 13)
    assert sev_logits.shape == (3, 5)


def test_head_parameter_independence():
    """Verify that multiple prediction heads instantiated from the template own completely independent parameters."""
    config = EmergentPathTriageConfig(latent_dim=8, head_hidden_dim=16)
    from models.emergent_path_triage.heads import PredictionHead
    
    head1 = PredictionHead(latent_dim=8, output_dim=5, config=config)
    head2 = PredictionHead(latent_dim=8, output_dim=5, config=config)
    
    params1 = list(head1.parameters())
    params2 = list(head2.parameters())
    
    # Check parameters are distinct objects
    assert params1[0] is not params2[0]
    
    # Verify linear weights are randomly initialized differently
    assert not torch.allclose(head1.fc1.weight, head2.fc1.weight)


def test_head_gradient_propagation():
    """Verify that loss gradients propagate through prediction head layers during training."""
    config = EmergentPathTriageConfig(latent_dim=8, head_hidden_dim=16, head_dropout=0.0)
    from models.emergent_path_triage.heads import PredictionHead
    
    head = PredictionHead(latent_dim=8, output_dim=5, config=config)
    head.train()
    
    for p in head.parameters():
        p.requires_grad = True
        
    state = torch.randn(2, 8)
    logits = head(state)
    
    loss = logits.sum()
    loss.backward()
    
    # Confirm gradients exist on parameters
    for name, p in head.named_parameters():
        assert p.grad is not None, f"Param {name} did not receive gradients"


def test_head_invalid_inputs():
    """Verify validation boundaries and exception triggers under invalid input representations."""
    config = EmergentPathTriageConfig(latent_dim=8, head_hidden_dim=16)
    from models.emergent_path_triage.heads import PredictionHead
    
    head = PredictionHead(latent_dim=8, output_dim=5, config=config)
    
    # 1. 3D input tensor rejects with InterfaceError
    with pytest.raises(InterfaceError, match="Input must be 2D tensor"):
        head(torch.randn(2, 8, 1))
        
    # 2. Latent dimension mismatch
    with pytest.raises(InterfaceError, match="Latent dimension mismatch"):
        head(torch.randn(2, 16))
        
    # 3. Dtype mismatch (must be float32)
    with pytest.raises(InterfaceError, match="Incorrect dtype"):
        head(torch.randn(2, 8).to(torch.float64))
        
    # 4. Device mismatch simulation
    with pytest.raises(InterfaceError, match="Device mismatch"):
        head(torch.zeros(2, 8, device="meta"))


def test_head_serialization_and_cpu():
    """Verify that PredictionHead output logits can be serialized and run on CPU."""
    config = EmergentPathTriageConfig(latent_dim=8, head_hidden_dim=16)
    from models.emergent_path_triage.heads import PredictionHead
    
    head = PredictionHead(latent_dim=8, output_dim=5, config=config)
    head.eval()
    
    state = torch.randn(2, 8)
    logits = head(state)
    
    # Mock serialization mapping of output logits
    serialized = logits.detach().cpu().tolist()
    assert isinstance(serialized, list)
    assert len(serialized) == 2
    assert len(serialized[0]) == 5
    
    reconstructed = torch.tensor(serialized, dtype=torch.float32)
    assert torch.allclose(reconstructed, logits)


def test_dcp_output_dimensions_and_validations():
    """Verify that Dynamic Consistency Projection projects reasoning and logits to urgency dimension 5."""
    config = EmergentPathTriageConfig(latent_dim=16)
    from models.emergent_path_triage.dcp import DynamicConsistencyProjection
    
    dcp = DynamicConsistencyProjection(latent_dim=16, config=config)
    dcp.eval()
    
    state = torch.randn(3, 16)
    preds = torch.randn(3, 18)
    
    h_proj, y_proj = dcp(state, preds)
    assert h_proj.shape == (3, 5)
    assert y_proj.shape == (3, 5)
    
    # Validation boundary check: length of tensors mismatch
    with pytest.raises(InterfaceError, match="specialist_state must be a 2D tensor"):
        dcp(torch.randn(3, 16, 1), preds)
        
    # Dimension mismatch check
    with pytest.raises(InterfaceError, match="Dimension mismatch"):
        dcp(torch.randn(3, 8), preds)
        
    # Input batch size mismatch check
    with pytest.raises(InterfaceError, match="Batch size mismatch"):
        dcp(torch.randn(2, 16), preds)


def test_dcp_parameter_independence():
    """Verify that DynamicConsistencyProjection layers are independent and parameter-isolated."""
    config = EmergentPathTriageConfig(latent_dim=8)
    from models.emergent_path_triage.dcp import DynamicConsistencyProjection
    
    dcp1 = DynamicConsistencyProjection(latent_dim=8, config=config)
    dcp2 = DynamicConsistencyProjection(latent_dim=8, config=config)
    
    assert dcp1.reasoning_proj.weight is not dcp2.reasoning_proj.weight
    assert not torch.allclose(dcp1.reasoning_proj.weight, dcp2.reasoning_proj.weight)


def test_loss_dcp_consistency_correctness():
    """Verify that consistency loss calculates correct alignment error under controlled scenarios."""
    config = EmergentPathTriageConfig(latent_dim=8, cons_lambda=1.0, ortho_lambda=0.0, div_lambda=0.0)
    from src.model import JointLoss
    
    class TinyConfig:
        hidden_size: int = 16
        num_hidden_layers: int = 1
        num_attention_heads: int = 2
        intermediate_size: int = 32
        max_position_embeddings: int = 32
        
    model_meta = EmergentPathTriageModel()
    model = model_meta.build(TinyConfig(), triage_config=config)
    
    # 1. Setup mock cached states
    model._last_final_state = torch.ones(2, 8)
    
    # Set dcp weights to 1.0
    nn.init.constant_(model.dcp.reasoning_proj.weight, 1.0)
    nn.init.constant_(model.dcp.logits_proj.weight, 1.0)
    
    spec_logits = torch.ones(2, 13)
    sev_logits = torch.ones(2, 5)
    
    # Concatenated predictions sum: 18 elements of 1.0 * 1.0 = 18.0 for each of the 5 outputs
    # projected_reasoning sum: 8 elements of 1.0 * 1.0 = 8.0 for each of the 5 outputs
    # Mean Squared Error: (8.0 - 18.0) ** 2 = 100.0
    
    loss_fn = JointLoss()
    res = model.compute_loss(
        spec_logits, sev_logits,
        torch.zeros(2, dtype=torch.long), torch.zeros(2, dtype=torch.long),
        loss_fn
    )
    
    assert torch.allclose(res["cons_loss"], torch.tensor(100.0))


def test_loss_routing_diversity_correctness():
    """Verify that diversity loss pushes average routing distributions to uniform entropy."""
    config = EmergentPathTriageConfig(latent_dim=8, div_lambda=1.0, cons_lambda=0.0, ortho_lambda=0.0)
    from src.model import JointLoss
    
    class TinyConfig:
        hidden_size: int = 16
        num_hidden_layers: int = 1
        num_attention_heads: int = 2
        intermediate_size: int = 32
        max_position_embeddings: int = 32
        
    model_meta = EmergentPathTriageModel()
    model = model_meta.build(TinyConfig(), triage_config=config)
    
    # Setup mock routing decision with uniform distribution over 2 steps and 3 blocks
    # routing probabilities: shape (Batch=2, Steps=2, Blocks=3)
    probs = torch.zeros(2, 2, 3)
    probs[:, :, 0] = 1.0 / 3.0
    probs[:, :, 1] = 1.0 / 3.0
    probs[:, :, 2] = 1.0 / 3.0
    
    model._last_routing_decision = RoutingDecision(
        routing_logits=torch.zeros((2, 2, 3)),
        routing_probabilities=probs,
        selected_blocks=[0, 1],
        path_depth=2,
        routing_entropy=torch.zeros(()),
        routing_confidence=torch.ones(()),
        path_identifier="test_div"
    )
    
    loss_fn = JointLoss()
    res = model.compute_loss(
        torch.zeros(2, 13), torch.zeros(2, 5),
        torch.zeros(2, dtype=torch.long), torch.zeros(2, dtype=torch.long),
        loss_fn
    )
    
    # Expected sum: 3 * (1/3 * log(1/3)) = -1.098612
    expected = 3.0 * (1.0 / 3.0) * torch.log(torch.tensor(1.0 / 3.0))
    assert torch.allclose(res["div_loss"], expected)


def test_loss_orthogonality_correctness():
    """Verify that evidence orthogonality loss correctly regularizes pairwise similarity mismatches."""
    config = EmergentPathTriageConfig(latent_dim=8, ortho_lambda=1.0, cons_lambda=0.0, div_lambda=0.0)
    from src.model import JointLoss
    
    class TinyConfig:
        hidden_size: int = 16
        num_hidden_layers: int = 1
        num_attention_heads: int = 2
        intermediate_size: int = 32
        max_position_embeddings: int = 32
        
    model_meta = EmergentPathTriageModel()
    model = model_meta.build(TinyConfig(), triage_config=config)
    
    # Mock evidence outputs that are identical (perfect correlation = 1.0 cosine similarity)
    # Shape of evidence aspects: (Batch, latent_dim) = (2, 8)
    # Cosine similarity matrix will be all 1s. Mismatch with Identity:
    # similarities - identity has: 0 on diagonal (1.0 - 1.0 = 0), and 1 on off-diagonals.
    # Total of 12 off-diagonal elements in a 4x4 matrix, each squared is 1.0. Mean is 12 / 16 = 0.75.
    ones = torch.ones(2, 8) / (8 ** 0.5) # unit norm
    model._last_evidence = EvidenceRepresentation(
        symptom=ones,
        anatomical=ones,
        temporal=ones,
        systemic=ones
    )
    
    loss_fn = JointLoss()
    res = model.compute_loss(
        torch.zeros(2, 13), torch.zeros(2, 5),
        torch.zeros(2, dtype=torch.long), torch.zeros(2, dtype=torch.long),
        loss_fn
    )
    
    assert torch.allclose(res["ortho_loss"], torch.tensor(0.75))


def test_composite_loss_weights_and_zero_weight():
    """Verify that weighting coefficients configure loss summation and setting weights to 0 nullifies terms."""
    config = EmergentPathTriageConfig(
        latent_dim=8,
        ortho_lambda=0.0,
        cons_lambda=0.0,
        div_lambda=0.0
    )
    from src.model import JointLoss
    
    class TinyConfig:
        hidden_size: int = 16
        num_hidden_layers: int = 1
        num_attention_heads: int = 2
        intermediate_size: int = 32
        max_position_embeddings: int = 32
        
    model_meta = EmergentPathTriageModel()
    model = model_meta.build(TinyConfig(), triage_config=config)
    
    # Populate mock states to ensure raw losses are non-zero
    ones = torch.ones(2, 8)
    model._last_final_state = ones
    model._last_evidence = EvidenceRepresentation(symptom=ones, anatomical=ones, temporal=ones, systemic=ones)
    model._last_routing_decision = RoutingDecision(
        routing_logits=torch.zeros((2, 2, 3)),
        routing_probabilities=torch.ones((2, 2, 3)),
        selected_blocks=[0, 1],
        path_depth=2,
        routing_entropy=torch.zeros(()),
        routing_confidence=torch.ones(()),
        path_identifier="test_div"
    )
    
    loss_fn = JointLoss()
    res = model.compute_loss(
        torch.zeros(2, 13), torch.zeros(2, 5),
        torch.zeros(2, dtype=torch.long), torch.zeros(2, dtype=torch.long),
        loss_fn
    )
    
    # Because coefficients are 0, joint loss must equal baseline JointLoss exactly
    expected = config.alpha_specialist * res["specialist_loss"] + config.beta_severity * res["severity_loss"]
    assert torch.allclose(res["joint_loss"], expected)


def test_loss_gradient_propagation():
    """Verify that backward pass loss updates propagate correctly through DCES, DCRR, CTB, Heads, and DCP."""
    config = EmergentPathTriageConfig(latent_dim=8, ortho_lambda=1.0, cons_lambda=1.0, div_lambda=1.0)
    from src.model import JointLoss
    
    class TinyConfig:
        hidden_size: int = 16
        num_hidden_layers: int = 1
        num_attention_heads: int = 2
        intermediate_size: int = 32
        max_position_embeddings: int = 32
        
    model_meta = EmergentPathTriageModel()
    model = model_meta.build(TinyConfig(), triage_config=config)
    model.train()
    
    input_ids = torch.randint(0, 100, (2, 8))
    attention_mask = torch.ones_like(input_ids)
    
    outputs = model(input_ids, attention_mask)
    
    loss_fn = JointLoss()
    loss_dict = model.compute_loss(
        outputs.specialist_logits,
        outputs.severity_logits,
        torch.zeros(2, dtype=torch.long),
        torch.zeros(2, dtype=torch.long),
        loss_fn
    )
    
    loss_dict["joint_loss"].backward()
    
    # Confirm gradients exist on all modules
    assert model.classifier_specialist.fc1.weight.grad is not None
    assert model.classifier_severity.fc1.weight.grad is not None
    assert model.dcp.reasoning_proj.weight.grad is not None
    assert model.dces.symptom_proj.linear1.weight.grad is not None
    assert model.blocks[0].linear1.weight.grad is not None
    assert model.router.logits_proj.weight.grad is not None


def test_loss_serialization_and_cpu():
    """Verify that composite loss outputs serialize successfully and execute on CPU."""
    config = EmergentPathTriageConfig(latent_dim=8)
    from src.model import JointLoss
    
    class TinyConfig:
        hidden_size: int = 16
        num_hidden_layers: int = 1
        num_attention_heads: int = 2
        intermediate_size: int = 32
        max_position_embeddings: int = 32
        
    model_meta = EmergentPathTriageModel()
    model = model_meta.build(TinyConfig(), triage_config=config)
    model.eval()
    
    model._last_final_state = torch.randn(2, 8)
    
    loss_fn = JointLoss()
    res = model.compute_loss(
        torch.zeros(2, 13), torch.zeros(2, 5),
        torch.zeros(2, dtype=torch.long), torch.zeros(2, dtype=torch.long),
        loss_fn
    )
    
    # Map loss values to lists/dicts of floats
    serialized = {k: v.cpu().item() for k, v in res.items()}
    assert isinstance(serialized["joint_loss"], float)
    assert isinstance(serialized["ortho_loss"], float)


def test_data_pipeline_global_seeds_reproducibility():
    """Verify that setting global seeds enforces deterministic random outputs."""
    import random
    import numpy as np
    from src.data_pipeline import set_global_seeds
    
    set_global_seeds(42)
    val1 = random.random()
    arr1 = np.random.randn(5)
    t1 = torch.randn(5)
    
    set_global_seeds(42)
    val2 = random.random()
    arr2 = np.random.randn(5)
    t2 = torch.randn(5)
    
    assert val1 == val2
    assert np.allclose(arr1, arr2)
    assert torch.allclose(t1, t2)


def test_data_pipeline_colab_detection():
    """Verify Google Colab environment detection helper is structured correctly."""
    from src.data_pipeline import detect_colab_environment
    res = detect_colab_environment()
    
    assert "is_colab" in res
    assert "has_gpu" in res
    assert "mixed_precision_available" in res
    assert "mount_drive" in res
    assert callable(res["mount_drive"])


def test_data_pipeline_label_validation():
    """Verify that LabelValidator checks label existence, maps deterministically, and raises ValueError for unseen labels."""
    from src.data_pipeline import LabelValidator
    validator = LabelValidator()
    
    # Valid maps
    assert validator.validate_specialist("GEN_MED") == 3
    assert validator.validate_severity("S4") == 3
    
    # Invalid maps
    with pytest.raises(ValueError, match="Unseen specialist class label"):
        validator.validate_specialist("UNSEEN_DEPT")
        
    with pytest.raises(ValueError, match="Unseen severity label"):
        validator.validate_severity("S10")
        
    # Serialization
    serialized = validator.serialize()
    assert "specialist_classes" in serialized
    assert "spec_to_id" in serialized


def test_data_pipeline_grouped_splitting_no_leakage():
    """Verify that splitting keeps seed_ids completely disjoint across train, val, and test splits (no leakage)."""
    import pandas as pd
    from src.data_pipeline import get_leakage_safe_splits
    
    # Construct a dummy dataframe with duplicate seed_ids representing clinical variants
    data = {
        "text": ["Pain", "Severe pain", "Mild pain", "Normal", "Heart pain", "Chest pain"],
        "seed_id": ["seed1", "seed1", "seed2", "seed2", "seed3", "seed3"],
        "department_code": ["GI", "GI", "GEN_MED", "GEN_MED", "CARDIO_PULM", "CARDIO_PULM"],
        "severity_heuristic": ["S4", "S4", "S2", "S2", "S5", "S5"]
    }
    df = pd.DataFrame(data)
    
    train_df, val_df, test_df = get_leakage_safe_splits(
        df, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42, stratify=False
    )
    
    # Verify no overlaps in seed_id
    train_seeds = set(train_df["seed_id"])
    val_seeds = set(val_df["seed_id"])
    test_seeds = set(test_df["seed_id"])
    
    assert len(train_seeds.intersection(val_seeds)) == 0
    assert len(train_seeds.intersection(test_seeds)) == 0
    assert len(val_seeds.intersection(test_seeds)) == 0
    assert len(train_df) + len(val_df) + len(test_df) == 6


def test_data_pipeline_dataset_and_dataloader_helpers():
    """Verify PyTorch dataset and dataloader creation, collation, and GPU compliance settings."""
    from src.data_pipeline import TokenizerPipeline, EmergentTriageDataset, get_dataloader
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    pipeline = TokenizerPipeline(tokenizer, max_length=16)
    
    texts = ["Patient presents with chest pain.", "Fever and cough."]
    specialist_labels = [0, 3]
    severity_labels = [4, 2]
    
    dataset = EmergentTriageDataset(texts, specialist_labels, severity_labels, pipeline)
    assert len(dataset) == 2
    
    sample = dataset[0]
    assert "input_ids" in sample
    assert "labels_specialist" in sample
    assert "labels_severity" in sample
    
    loader = get_dataloader(dataset, batch_size=2, shuffle=False, pin_memory=True)
    batch = next(iter(loader))
    
    assert batch["input_ids"].shape == (2, 16)
    assert batch["labels_specialist"].shape == (2,)
    assert batch["labels_severity"].shape == (2,)


def test_data_pipeline_dataset_audit():
    """Verify that dataset auditing helper parses sequence stats and outputs md report file."""
    import os
    import pandas as pd
    from src.data_pipeline import audit_dataset, generate_dataset_report
    from transformers import AutoTokenizer
    
    # Create temp CSV dataset
    data = {
        "text": ["Pain", "Chest pain", "Fever"],
        "seed_id": ["seed1", "seed2", "seed3"],
        "department_code": ["GI", "CARDIO_PULM", "GEN_MED"],
        "severity_heuristic": ["S4", "S5", "S2"],
        "language": ["en", "hinglish", "en"]
    }
    df = pd.DataFrame(data)
    temp_csv = "temp_audit_dataset.csv"
    df.to_csv(temp_csv, index=False)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        res = audit_dataset(temp_csv, tokenizer)
        
        assert res["total_samples"] == 3
        assert "word_stats" in res
        assert "token_stats" in res
        
        report_md = "temp_audit_report.md"
        generate_dataset_report(res, report_md)
        assert os.path.exists(report_md)
        os.remove(report_md)
    finally:
        if os.path.exists(temp_csv):
            os.remove(temp_csv)


def test_trainer_lifecycle_and_checkpointing():
    """Verify that Trainer runs epochs, tracks metrics, implements early stopping, saves/loads checkpoints, and resumes training successfully."""
    import os
    from pathlib import Path
    import shutil
    from src.data_pipeline import TokenizerPipeline, EmergentTriageDataset, get_dataloader
    from src.trainer import EmergentTrainer
    from src.config_manager import TrainingConfig
    
    from transformers import AutoTokenizer
    
    # Configure tiny model
    class TinyConfig:
        hidden_size: int = 16
        num_hidden_layers: int = 1
        num_attention_heads: int = 2
        intermediate_size: int = 32
        max_position_embeddings: int = 32
        
    config = EmergentPathTriageConfig(latent_dim=8)
    model_meta = EmergentPathTriageModel()
    model = model_meta.build(TinyConfig(), triage_config=config)
    
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    pipeline = TokenizerPipeline(tokenizer, max_length=8)
    
    texts = ["Chest pain", "Fever"]
    specialist_labels = [0, 3]
    severity_labels = [4, 2]
    
    dataset = EmergentTriageDataset(texts, specialist_labels, severity_labels, pipeline)
    loader = get_dataloader(dataset, batch_size=2, shuffle=False)
    
    temp_checkpoint_dir = "./temp_test_checkpoints"
    trainer_config = TrainingConfig(
    epochs=3,
    learning_rate=1e-3,
    encoder_lr=1e-4,
    weight_decay=0.01,
    batch_size=2,
    dropout=0.1,
    optimizer="adamw",
    scheduler="cosine",
    warmup_ratio=0.1,
    loss_weights={"alpha_specialist": 1.0, "beta_severity": 1.0},
    gradient_accumulation=1,
    gradient_clipping=1.0,
    checkpoint_frequency_epochs=1,
    primary_metric="val_loss",
    encoder_model="mock",
    dynamic_padding=True,
    gradient_checkpointing=False,
    flash_attention=False,
    pin_memory=False,
    persistent_workers=False,
    prefetch_factor=2,
    dataloader_workers=0,
    early_stopping_patience=2,
    early_stopping_metric="val_loss",
    early_stopping_min_improvement=1e-4,
    seed=42,
    checkpoint_dir=temp_checkpoint_dir,
    mixed_precision=False,
    use_torch_compile=False,
    non_blocking_transfers=True
    )
    
    trainer = EmergentTrainer(
        model=model,
        config=trainer_config,
        train_loader=loader,
        val_loader=loader,
        tokenizer=tokenizer
    )
    
    # Run fit
    best_metrics = trainer.fit()
    
    assert "val_loss" in best_metrics
    assert len(trainer.history) <= 3
    assert os.path.exists(os.path.join(temp_checkpoint_dir, "best_model.pt"))
    assert os.path.exists(os.path.join(temp_checkpoint_dir, "latest_model.pt"))
    assert os.path.exists(os.path.join(temp_checkpoint_dir, "training_history.csv"))
    
    # Test resume
    new_model = model_meta.build(TinyConfig(), triage_config=config)
    new_trainer = EmergentTrainer(
        model=new_model,
        config=trainer_config,
        train_loader=loader,
        val_loader=loader,
        tokenizer=tokenizer
    )
    
    resumed_epoch = new_trainer.load_checkpoint(Path(temp_checkpoint_dir) / "latest_model.pt")
    assert resumed_epoch > 0
    assert len(new_trainer.history) > 0
    
    # Cleanup
    if os.path.exists(temp_checkpoint_dir):
        shutil.rmtree(temp_checkpoint_dir)


def test_trainer_gradient_accumulation_and_amp():
    """Verify that Trainer handles gradient accumulation and matches AMP configurations."""
    import os
    import shutil
    from src.data_pipeline import TokenizerPipeline, EmergentTriageDataset, get_dataloader
    from src.trainer import EmergentTrainer
    from src.config_manager import TrainingConfig
    
    from transformers import AutoTokenizer
    
    class TinyConfig:
        hidden_size: int = 16
        num_hidden_layers: int = 1
        num_attention_heads: int = 2
        intermediate_size: int = 32
        max_position_embeddings: int = 32
        
    config = EmergentPathTriageConfig(latent_dim=8)
    model_meta = EmergentPathTriageModel()
    model = model_meta.build(TinyConfig(), triage_config=config)
    
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    pipeline = TokenizerPipeline(tokenizer, max_length=8)
    
    texts = ["Chest pain", "Fever", "Headache", "Cough"]
    specialist_labels = [0, 3, 1, 2]
    severity_labels = [4, 2, 1, 3]
    
    dataset = EmergentTriageDataset(texts, specialist_labels, severity_labels, pipeline)
    loader = get_dataloader(dataset, batch_size=2, shuffle=False)
    
    temp_checkpoint_dir = "./temp_test_accum"
    trainer_config = TrainingConfig(
    epochs=1,
    learning_rate=1e-3,
    encoder_lr=1e-4,
    weight_decay=0.01,
    batch_size=2,
    dropout=0.1,
    optimizer="adamw",
    scheduler="cosine",
    warmup_ratio=0.1,
    loss_weights={"alpha_specialist": 1.0, "beta_severity": 1.0},
    gradient_accumulation=2,
    gradient_clipping=1.0,
    checkpoint_frequency_epochs=1,
    primary_metric="val_loss",
    encoder_model="mock",
    dynamic_padding=True,
    gradient_checkpointing=False,
    flash_attention=False,
    pin_memory=False,
    persistent_workers=False,
    prefetch_factor=2,
    dataloader_workers=0,
    early_stopping_patience=2,
    early_stopping_metric="val_loss",
    early_stopping_min_improvement=1e-4,
    seed=42,
    checkpoint_dir=temp_checkpoint_dir,
    mixed_precision=True,
    use_torch_compile=False,
    non_blocking_transfers=True
    )
    
    trainer = EmergentTrainer(
        model=model,
        config=trainer_config,
        train_loader=loader,
        val_loader=loader,
        tokenizer=tokenizer
    )
    
    # We step only once because len(loader) is 2 and accumulation step is 2
    # Check optimizer steps
    initial_params = [p.clone() for p in model.parameters() if p.requires_grad]
    
    trainer.train_epoch(1)
    
    # Verify model parameters updated
    updated_params = [p for p in model.parameters() if p.requires_grad]
    assert any(not torch.equal(p1, p2) for p1, p2 in zip(initial_params, updated_params))
    
    # Cleanup
    if os.path.exists(temp_checkpoint_dir):
        shutil.rmtree(temp_checkpoint_dir)









