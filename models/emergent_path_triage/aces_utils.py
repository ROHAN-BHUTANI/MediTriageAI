"""Analysis and diagnostic utilities for the Adaptive Clinical Evidence Synthesizer (ACES)."""

from typing import Any

import torch

from models.emergent_path_triage.types import EvidenceReasoningTrace


class EvidenceDiagnostics:
    """Experiment-ready diagnostics and metric aggregation for ACES."""

    @staticmethod
    def aggregate_statistics(trace: EvidenceReasoningTrace) -> dict[str, Any]:
        """Aggregate statistical metrics from a reasoning trace without affecting inference.

        Supports:
        - Average attention entropy
        - Per-class aspect importance
        - Prototype utilization
        - Attention head diversity
        - Sparsity statistics
        - Evidence confidence statistics
        """
        if trace.fusion_type != "AttentionFusion":
            return {
                "fusion_type": trace.fusion_type,
                "message": "Detailed diagnostics require AttentionFusion",
            }

        stats = {}

        # 1. Average attention entropy
        stats["average_attention_entropy"] = trace.fusion_entropy

        # 2. Per-class aspect importance
        if trace.aspect_importance_scores is not None:
            importance = {}
            for k, v in trace.aspect_importance_scores.items():
                importance[k] = float(v.mean().item())
            stats["per_class_aspect_importance"] = importance

            # 5. Sparsity statistics (percentage of times an aspect is strongly suppressed)
            sparsity = {}
            threshold = 0.05
            for k, v in trace.aspect_importance_scores.items():
                sparsity[k] = float((v < threshold).float().mean().item())
            stats["sparsity_statistics"] = sparsity

            # 6. Evidence confidence statistics (mean max importance across aspects)
            # Higher confidence implies the model heavily relies on at least one aspect
            stacked = torch.stack(list(trace.aspect_importance_scores.values()), dim=1)
            stats["evidence_confidence_mean"] = float(
                stacked.max(dim=1).values.mean().item()
            )

        # 3. Prototype utilization
        stats["prototype_utilization"] = trace.prototype_utilization

        # 4. Attention head diversity
        if trace.interaction_weights is not None:
            # interaction_weights shape: (Batch, Num_Heads, 4, 4)
            # Flatten 4x4 attention matrix for each head to shape (Batch, Num_Heads, 16)
            B, H, _, _ = trace.interaction_weights.shape
            if H > 1:
                flat_weights = trace.interaction_weights.view(B, H, 16)
                norms = torch.norm(flat_weights, p=2, dim=2, keepdim=True).clamp(
                    min=1e-8
                )
                normalized = flat_weights / norms
                # Cosine similarity between heads
                sim = torch.bmm(normalized, normalized.transpose(1, 2))  # (Batch, H, H)
                # Average non-diagonal similarities
                mask = ~torch.eye(H, dtype=torch.bool, device=sim.device)
                div = []
                for b in range(B):
                    div.append(sim[b][mask].mean().item())
                # Lower cosine sim = higher diversity.
                stats["attention_head_similarity"] = sum(div) / B if B > 0 else 0.0
            else:
                stats["attention_head_similarity"] = 1.0

        return stats
