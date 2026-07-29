"""Clinical Thought Execution Engine for E-PATH-CO-REASON.

Contains the single-step ClinicalThoughtExecutionEngine and a backward-compatible
factory function that exposes the old ReasoningPathExecutionEngine name via the
LegacyExecutionEngineAdapter.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.emergent_path_triage.config import EmergentPathTriageConfig
from models.emergent_path_triage.exceptions import InterfaceError
from models.emergent_path_triage.logger import get_logger
from models.emergent_path_triage.types import ExecutionInstruction

logger = get_logger()


class ClinicalThoughtExecutionEngine(nn.Module):
    """Single-step thought block executor.
    
    Receives an ExecutionInstruction and returns the updated reasoning state.
    Has zero knowledge of routing semantics, router state, or trace recording.
    
    ============================================================================
    MATH CORRESPONDENCE
    ============================================================================
    - Training Mode (differentiable soft blend):
        h_{t+1} = sum_{j=1}^N w_{t,j} * CTB_j(h_t)
      where w_{t,j} are the execution weights from the instruction.
    - Inference Mode (hard conditional execution):
        h_{t+1} = CTB_{k_t}(h_t)
      where k_t = selected_blocks[b] for each batch sample b.
      
    ============================================================================
    COMPUTATIONAL COMPLEXITY
    ============================================================================
    Time Complexity:
    - Inference: O(B * d * H_ctb) -> Executes exactly one block per sample.
    - Training:  O(B * N * d * H_ctb) -> Executes all N blocks for gradient blending.
    Space Complexity:
    - O(B * N * d) during training for stacked block outputs.
    """

    def __init__(self, config: EmergentPathTriageConfig) -> None:
        super().__init__()
        self.config = config
        logger.info(
            f"Initialized ClinicalThoughtExecutionEngine with "
            f"num_thought_blocks={config.num_thought_blocks}"
        )

    def _execute_block(self, block_idx: int, state: torch.Tensor, blocks: nn.ModuleList) -> torch.Tensor:
        """Execute a single thought block with ablation-aware bypass.

        Args:
            block_idx: Index of the block to execute.
            state: (Batch, Latent_Dim) current reasoning state.
            blocks: Full collection of ClinicalThoughtBlocks.
        Returns:
            Updated state (Batch, Latent_Dim).
        """
        enabled = True
        if block_idx == 0:
            enabled = getattr(self.config, "ablation_ctb1_enabled", True)
        elif block_idx == 1:
            enabled = getattr(self.config, "ablation_ctb2_enabled", True)
        elif block_idx == 2:
            enabled = getattr(self.config, "ablation_ctb3_enabled", True)
        elif block_idx == 3:
            enabled = getattr(self.config, "ablation_ctb4_enabled", True)

        if enabled:
            return blocks[block_idx](state)
        return state

    def forward(
        self,
        current_state: torch.Tensor,
        instruction: ExecutionInstruction,
        blocks: nn.ModuleList,
    ) -> torch.Tensor:
        """Execute one reasoning step.
        
        Args:
            current_state: (Batch, Latent_Dim) current reasoning representation h_t.
            instruction: ExecutionInstruction produced by Router.
            blocks: The full set of ClinicalThoughtBlocks.
        Returns:
            next_state: (Batch, Latent_Dim) updated reasoning representation h_{t+1}.
        """
        if not isinstance(instruction, ExecutionInstruction):
            raise InterfaceError(
                f"ClinicalThoughtExecutionEngine expects an ExecutionInstruction, got {type(instruction)}"
            )

        num_blocks = len(blocks)

        if self.training:
            # Differentiable soft blend: execute all blocks and weight
            updated_states = [self._execute_block(i, current_state, blocks) for i in range(num_blocks)]
            stacked = torch.stack(updated_states, dim=1)     # (B, N, d)
            weights = instruction.execution_weights           # (B, N)
            next_state = torch.sum(stacked * weights.unsqueeze(-1), dim=1)
        else:
            # Hard routing: execute all blocks and gather (efficient for small N)
            updated_states = [self._execute_block(i, current_state, blocks) for i in range(num_blocks)]
            stacked = torch.stack(updated_states, dim=1)     # (B, N, d)
            idx = instruction.selected_blocks.unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
            idx = idx.expand(-1, 1, stacked.size(-1))        # (B, 1, d)
            next_state = stacked.gather(1, idx).squeeze(1)   # (B, d)

        return next_state


def ReasoningPathExecutionEngine(config: EmergentPathTriageConfig) -> nn.Module:
    """Factory function preserving the old constructor signature.
    
    Returns a LegacyExecutionEngineAdapter wrapping a ClinicalThoughtExecutionEngine,
    providing full backward compatibility with the old forward(evidence_list, routing_decision, blocks) API.
    
    Args:
        config: EmergentPathTriageConfig instance.
    Returns:
        A LegacyExecutionEngineAdapter instance.
    """
    from models.emergent_path_triage.compat import LegacyExecutionEngineAdapter
    step_engine = ClinicalThoughtExecutionEngine(config)
    adapter = LegacyExecutionEngineAdapter(step_engine, config)
    
    # Register the ExecutionEngineAuditor for observability (legacy compatibility)
    from models.emergent_path_triage.hooks import ExecutionEngineAuditor
    adapter.auditor = ExecutionEngineAuditor(adapter)
    
    logger.info(
        f"Created ReasoningPathExecutionEngine (legacy adapter) with max_path_depth={config.max_path_depth}, "
        f"num_thought_blocks={config.num_thought_blocks}"
    )
    return adapter
