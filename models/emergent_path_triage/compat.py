"""Legacy compatibility adapters for E-PATH-CO-REASON.

Provides backward-compatible wrappers around the new single-step
ClinicalThoughtExecutionEngine for consumers still using the old
multi-step ReasoningPathExecutionEngine API.
"""

from __future__ import annotations

from pathlib import Path
import torch
import torch.nn as nn

from models.emergent_path_triage.config import EmergentPathTriageConfig
from models.emergent_path_triage.exceptions import InterfaceError, RoutingError
from models.emergent_path_triage.interfaces import BaseClinicalThoughtBlock
from models.emergent_path_triage.types import (
    ExecutionInstruction,
    RoutingDecision,
    ThoughtPath,
)


class LegacyExecutionEngineAdapter(nn.Module):
    """Wraps ClinicalThoughtExecutionEngine to expose the old
    ReasoningPathExecutionEngine.forward(evidence_list, routing_decision, blocks)
    signature for existing tests and scripts.
    """

    def __init__(self, step_engine: nn.Module, config: EmergentPathTriageConfig, evidence_projection: nn.Module) -> None:
        super().__init__()
        self.step_engine = step_engine
        self.config = config
        self.evidence_projection = evidence_projection

    def forward(
        self,
        evidence_list: list[torch.Tensor],
        routing_decision: RoutingDecision,
        blocks: nn.ModuleList,
    ) -> tuple[torch.Tensor, ThoughtPath]:
        """Implement legacy multi-step loop using single-step engine."""
        # Validate inputs
        if not isinstance(routing_decision, RoutingDecision):
            raise InterfaceError(
                f"LegacyExecutionEngineAdapter expects a RoutingDecision, got {type(routing_decision)}"
            )
        if len(evidence_list) != 4:
            raise InterfaceError(f"Expected exactly 4 aspect evidence tensors, got {len(evidence_list)}")

        device = evidence_list[0].device
        batch_size = evidence_list[0].shape[0]

        # Verify blocks collection
        if len(blocks) != self.config.num_thought_blocks:
            raise InterfaceError(
                f"Blocks count mismatch: config specifies {self.config.num_thought_blocks} "
                f"but received {len(blocks)} blocks"
            )
        for idx, block in enumerate(blocks):
            if not isinstance(block, BaseClinicalThoughtBlock):
                raise InterfaceError(f"Block at index {idx} does not inherit from BaseClinicalThoughtBlock")

        # Verify routing bounds
        if routing_decision.path_depth != self.config.max_path_depth:
            raise RoutingError(
                f"Routing path depth mismatch: DCRR produced depth {routing_decision.path_depth} "
                f"but config requires {self.config.max_path_depth}"
            )
        for idx, step_idx in enumerate(routing_decision.selected_blocks):
            if not (0 <= step_idx < self.config.num_thought_blocks):
                raise RoutingError(
                    f"Invalid block selection: Selected index {step_idx} at step {idx} "
                    f"is out of bounds for {self.config.num_thought_blocks} blocks"
                )

        # Initialize h_0
        fused = torch.cat(evidence_list, dim=-1)
        h_t = self.evidence_projection(fused)
        representations = [h_t]

        # Configuration-driven ablation bypass
        if not getattr(self.config, "ablation_engine_enabled", True):
            depth = routing_decision.path_depth
            representations = [h_t] * (depth + 1)
            thought_path = ThoughtPath(
                states=routing_decision.selected_blocks,
                representations=representations,
            )
            return h_t, thought_path

        # Determine depth (ablation override)
        depth = self.config.max_path_depth
        if not getattr(self.config, "ablation_multistep_enabled", True):
            depth = 1

        for m in range(depth):
            selected_block = routing_decision.selected_blocks[m]
            step_probs = routing_decision.routing_probabilities[:, m, :]

            if hasattr(self, "auditor") and self.auditor is not None:
                self.auditor.set_current_step(m)

            instruction = ExecutionInstruction(
                selected_blocks=torch.tensor(
                    [selected_block] * batch_size,
                    dtype=torch.int64, device=device,
                ),
                execution_weights=step_probs,
            )
            h_t = self.step_engine(h_t, instruction, blocks)
            representations.append(h_t)

        thought_path = ThoughtPath(
            states=routing_decision.selected_blocks[:depth],
            representations=representations,
        )
        return h_t, thought_path

    def reset_audit(self) -> None:
        """Reset execution engine auditor data."""
        if hasattr(self, "auditor") and self.auditor is not None:
            self.auditor.reset()

    def finalize_and_export_audit(
        self,
        model: nn.Module,
        last_batch: dict | None,
        device: torch.device,
        use_amp: bool,
        checkpoint_dir: str | Path | None,
    ) -> None:
        """Run backward pass audit on the last batch and write files."""
        if hasattr(self, "auditor") and self.auditor is not None:
            self.auditor.finalize_and_export_audit(
                model=model,
                last_batch=last_batch,
                device=device,
                use_amp=use_amp,
                checkpoint_dir=checkpoint_dir,
            )
