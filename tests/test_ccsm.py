"""CCSM specific unit tests for E-PATH-CO-REASON."""

import torch
import torch.nn as nn

from models.emergent_path_triage.config import EmergentPathTriageConfig
from models.emergent_path_triage.types import (
    TraceRecordingLevel,
    TraceRecordingConfig,
    TraceRecorder,
)
from models.emergent_path_triage.dcrr import ClinicalReasoningRouter
from models.emergent_path_triage.engine import ClinicalThoughtExecutionEngine
from models.emergent_path_triage.ctb import ClinicalThoughtBlock


def test_trace_recording_config_resolution():
    """Verify trace config resolves flags by priority."""
    # Default STANDARD config
    cfg = TraceRecordingConfig(level=TraceRecordingLevel.STANDARD)
    assert cfg.should_record("entropy") is True
    assert cfg.should_record("probabilities") is True
    assert cfg.should_record("logits") is False
    assert cfg.should_record("hidden_states") is False

    # Override level with explicit flag
    cfg = TraceRecordingConfig(
        level=TraceRecordingLevel.MINIMAL,
        record_logits=True
    )
    assert cfg.should_record("logits") is True
    assert cfg.should_record("probabilities") is False

    # FULL records everything by default
    cfg = TraceRecordingConfig(level=TraceRecordingLevel.FULL)
    assert cfg.should_record("hidden_states") is True


def test_ccsm_router_step():
    """Verify CCSM Router initializes state and steps through GRU correctly."""
    config = EmergentPathTriageConfig(
        latent_dim=8,
        routing_hidden_dim=16,
        num_thought_blocks=4,
        max_path_depth=3
    )
    router = ClinicalReasoningRouter(config)
    
    batch_size = 2
    fused_evidence = torch.randn(batch_size, 4 * 8)
    
    # Init state
    state = router.init_state(fused_evidence)
    assert state.step_index == 0
    assert state.hidden_state.shape == (batch_size, 16)
    assert len(state.routing_history) == 0
    
    # Step
    h_t = torch.randn(batch_size, 8)
    out = router.step(h_t, state, temperature=1.0)
    
    assert out.next_router_state.step_index == 1
    assert out.next_router_state.hidden_state.shape == (batch_size, 16)
    assert not torch.allclose(state.hidden_state, out.next_router_state.hidden_state)
    assert out.routing_logits.shape == (batch_size, 4)
    assert out.selected_blocks.shape == (batch_size,)


def test_ccsm_execution_engine():
    """Verify the single-step Execution Engine processes instructions correctly."""
    config = EmergentPathTriageConfig(latent_dim=8, num_thought_blocks=4)
    engine = ClinicalThoughtExecutionEngine(config)
    blocks = nn.ModuleList([
        ClinicalThoughtBlock(8, config) for _ in range(4)
    ])
    
    batch_size = 2
    h_t = torch.randn(batch_size, 8)
    
    # Instruction selecting block 2
    from models.emergent_path_triage.types import ExecutionInstruction
    instruction = ExecutionInstruction(
        selected_blocks=torch.tensor([2, 2]),
        execution_weights=torch.zeros(batch_size, 4)
    )
    # Set probability 1.0 for block 2
    instruction.execution_weights[:, 2] = 1.0
    
    # Eval mode for both to avoid dropout differences
    engine.eval()
    for b in blocks:
        b.eval()
        
    # Training (soft blend, but in eval mode for deterministic check)
    engine.training = True # manually override just to force soft execution
    h_next_soft = engine(h_t, instruction, blocks)
    assert h_next_soft.shape == (batch_size, 8)
    
    # Eval (hard gather)
    engine.training = False # force hard execution
    h_next_hard = engine(h_t, instruction, blocks)
    assert h_next_hard.shape == (batch_size, 8)
    
    # Soft and hard should exactly match when weights are 1-hot
    assert torch.allclose(h_next_soft, h_next_hard, atol=1e-5)


def test_ccsm_trace_to_legacy_decision():
    """Verify trace fallback missing data maps to correct shapes."""
    config = EmergentPathTriageConfig(
        latent_dim=8,
        routing_trace_level="STANDARD" # No logits
    )
    router = ClinicalReasoningRouter(config)
    
    batch_size = 1
    fused_evidence = torch.randn(batch_size, 4 * 8)
    state = router.init_state(fused_evidence)
    h_t = torch.randn(batch_size, 8)
    
    step_out = router.step(h_t, state, 1.0)
    
    trace_cfg = TraceRecordingConfig(level=TraceRecordingLevel.STANDARD)
    recorder = TraceRecorder(trace_cfg)
    recorder.reset()
    recorder.record(step_out, h_t)
    trace = recorder.finalize("test_path", 1)
    
    decision = trace.to_routing_decision(device="cpu")
    
    # Probabilities should match
    assert decision.routing_probabilities.shape == (1, 1, 4)
    # Logits should be padded to 0 with matching shape
    assert decision.routing_logits.shape == (1, 1, 4)
    assert torch.all(decision.routing_logits == 0)
