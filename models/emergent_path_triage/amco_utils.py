import torch

from .types import OptimizationReasoningTrace


class OptimizationDiagnostics:
    """
    Computes experiment-ready metrics from the OptimizationReasoningTrace.
    """

    @staticmethod
    def aggregate_statistics(trace: OptimizationReasoningTrace) -> dict[str, float]:
        """
        Calculates diagnostic metrics based on the trace.
        Metrics:
        - effective task weights
        - uncertainty evolution
        - task imbalance index
        - optimization entropy
        - gradient magnitude statistics (if available)
        - convergence stability (delta from previous if available)
        """
        metrics = {}

        if not trace:
            return metrics

        # Optimization Entropy (How uniform are the weights?)
        if trace.effective_task_weights:
            weights = list(trace.effective_task_weights.values())
            if len(weights) > 0:
                w_tensor = torch.stack([w.cpu() for w in weights]).squeeze()
                # Normalize weights to form a probability distribution for entropy
                if w_tensor.sum() > 0:
                    p = w_tensor / w_tensor.sum()
                    entropy = -torch.sum(p * torch.log(p + 1e-9)).item()
                    metrics["optimization_entropy"] = entropy

                # Task Imbalance Index (Variance of normalized weights)
                imbalance = torch.var(w_tensor).item()
                metrics["task_imbalance_index"] = imbalance

                # Log the effective weights as well
                for task_name, w in trace.effective_task_weights.items():
                    metrics[f"weight_{task_name}"] = w.item()

        # Uncertainty Evolution
        if trace.uncertainty_estimates:
            for task_name, u in trace.uncertainty_estimates.items():
                metrics[f"uncertainty_{task_name}"] = u.item()

        # Gradient Magnitude Statistics
        if trace.gradient_statistics:
            for task_name, g in trace.gradient_statistics.items():
                metrics[f"grad_mag_{task_name}"] = g.item()

        # Task Convergence Metrics
        if trace.task_convergence_metrics:
            for task_name, c in trace.task_convergence_metrics.items():
                metrics[f"convergence_{task_name}"] = c

        return metrics
