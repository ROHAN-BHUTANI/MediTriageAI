import torch
import torch.nn as nn
from typing import Any
from abc import ABC, abstractmethod

from .config import EmergentPathTriageConfig
from .types import OptimizationReasoningTrace, OptimizationRecorder


class BaseLossBalancer(nn.Module, ABC):
    """
    Abstract base interface for Adaptive Multi-Task Clinical Optimization (AMCO).
    Enforces a strict multi-stage optimization pipeline.
    """
    
    def __init__(self, config: EmergentPathTriageConfig, task_names: list[str]) -> None:
        super().__init__()
        self.config = config
        self.task_names = task_names
        self.recorder = OptimizationRecorder()

    def forward(self, losses: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Executes the optimization pipeline.
        
        Pipeline:
        1. Task Loss Collection
        2. Task Statistics Extraction
        3. Balancing Strategy
        4. Weight Generation
        5. Composite Loss Assembly
        6. Optimization Telemetry
        """
        # 1. Task Loss Collection
        loss_tensors = self._collect_losses(losses)
        
        # 2. Task Statistics Extraction
        stats = self._extract_statistics(loss_tensors)
        
        # 3. Balancing Strategy
        strategy_output = self._apply_strategy(loss_tensors, stats)
        
        # 4. Weight Generation
        weights, regularization = self._generate_weights(strategy_output)
        
        # 5. Composite Loss Assembly
        total_loss = self._assemble_loss(loss_tensors, weights, regularization)
        
        # 6. Optimization Telemetry
        self._record_telemetry(weights, strategy_output)
        
        # Return total loss and scalar weights for basic logging
        weight_scalars = {t: weights[t].item() for t in self.task_names}
        return total_loss, weight_scalars

    def _collect_losses(self, losses: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        for task in self.task_names:
            if task not in losses:
                raise ValueError(f"Expected loss for task '{task}', but missing from input.")
        return {task: losses[task] for task in self.task_names}

    @abstractmethod
    def _extract_statistics(self, loss_tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def _apply_strategy(self, loss_tensors: dict[str, torch.Tensor], stats: dict[str, torch.Tensor]) -> dict[str, Any]:
        pass

    @abstractmethod
    def _generate_weights(self, strategy_output: dict[str, Any]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Returns (weights_dict, regularization_scalar)."""
        pass

    def _assemble_loss(self, loss_tensors: dict[str, torch.Tensor], weights: dict[str, torch.Tensor], regularization: torch.Tensor) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=next(iter(loss_tensors.values())).device)
        for task in self.task_names:
            total_loss = total_loss + weights[task] * loss_tensors[task]
        total_loss = total_loss + regularization
        return total_loss

    @abstractmethod
    def _record_telemetry(self, weights: dict[str, torch.Tensor], strategy_output: dict[str, Any]) -> None:
        pass


class StaticLossBalancer(BaseLossBalancer):
    """
    Legacy baseline implementing fixed static weights.
    """
    def __init__(self, config: EmergentPathTriageConfig, task_names: list[str]) -> None:
        super().__init__(config, task_names)
        
        # Setup fixed weights based on config
        self.fixed_weights = {}
        for task in task_names:
            # Fallbacks for specific tasks if needed, e.g. specialist -> alpha, severity -> beta
            # But normally we just use config values if available
            weight = 1.0
            if task == "specialist":
                weight = config.alpha_specialist
            elif task == "severity":
                weight = config.beta_severity
            elif task == "ortho":
                weight = config.ortho_lambda
            elif task == "cons":
                weight = config.cons_lambda
            elif task == "div":
                weight = config.div_lambda
            self.fixed_weights[task] = float(weight)

    def _extract_statistics(self, loss_tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {}

    def _apply_strategy(self, loss_tensors: dict[str, torch.Tensor], stats: dict[str, torch.Tensor]) -> dict[str, Any]:
        return {}

    def _generate_weights(self, strategy_output: dict[str, Any]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        device = next(iter(strategy_output.values())).device if strategy_output else "cpu"
        weights = {task: torch.tensor(self.fixed_weights[task], device=device) for task in self.task_names}
        reg = torch.tensor(0.0, device=device)
        return weights, reg
        
    def _assemble_loss(self, loss_tensors: dict[str, torch.Tensor], weights: dict[str, torch.Tensor], regularization: torch.Tensor) -> torch.Tensor:
        # Override to use the exact devices of the loss tensors to avoid cpu/cuda mismatches
        total_loss = None
        for task in self.task_names:
            device = loss_tensors[task].device
            w = torch.tensor(self.fixed_weights[task], device=device)
            weighted_loss = w * loss_tensors[task]
            if total_loss is None:
                total_loss = weighted_loss
            else:
                total_loss = total_loss + weighted_loss
        return total_loss

    def _record_telemetry(self, weights: dict[str, torch.Tensor], strategy_output: dict[str, Any]) -> None:
        if self.recorder.record_enabled:
            trace = OptimizationReasoningTrace(
                optimization_type="STATIC",
                effective_task_weights={k: v.detach().clone() for k, v in weights.items()}
            )
            self.recorder.record(trace)


class HomoscedasticBalancer(BaseLossBalancer):
    """
    Homoscedastic Uncertainty loss balancer.
    L = sum_i [ exp(-s_i) * L_i + 0.5 * s_i ]
    """
    def __init__(self, config: EmergentPathTriageConfig, task_names: list[str]) -> None:
        super().__init__(config, task_names)
        self.log_vars = nn.ParameterDict({
            task: nn.Parameter(torch.zeros(1)) for task in task_names
        })

    def _extract_statistics(self, loss_tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {}

    def _apply_strategy(self, loss_tensors: dict[str, torch.Tensor], stats: dict[str, torch.Tensor]) -> dict[str, Any]:
        return {"log_vars": self.log_vars}

    def _generate_weights(self, strategy_output: dict[str, Any]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        log_vars = strategy_output["log_vars"]
        weights = {}
        reg = None
        
        for task in self.task_names:
            s_i = log_vars[task]
            weights[task] = torch.exp(-s_i)
            task_reg = 0.5 * s_i
            
            if reg is None:
                reg = task_reg
            else:
                reg = reg + task_reg
                
        return weights, reg

    def _record_telemetry(self, weights: dict[str, torch.Tensor], strategy_output: dict[str, Any]) -> None:
        if self.recorder.record_enabled:
            log_vars = strategy_output["log_vars"]
            uncertainties = {task: log_vars[task].detach().clone() for task in self.task_names}
            
            trace = OptimizationReasoningTrace(
                optimization_type="HOMOSCEDASTIC",
                effective_task_weights={k: v.detach().clone() for k, v in weights.items()},
                uncertainty_estimates=uncertainties
            )
            self.recorder.record(trace)

class GradNormBalancer(BaseLossBalancer):
    """
    GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks.
    Dynamically balances task losses to ensure all tasks train at similar rates.
    """
    def __init__(self, config: EmergentPathTriageConfig, task_names: list[str], shared_param: torch.nn.Parameter | None = None) -> None:
        super().__init__(config, task_names)
        self.shared_param = shared_param
        self.register_buffer("w_i", torch.ones(len(task_names)))
        self.register_buffer("initial_losses", torch.zeros(len(task_names)))
        self.alpha = 1.5
        self.lr = 0.025
        self.step_count = 0

    def _extract_statistics(self, loss_tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {}

    def _apply_strategy(self, loss_tensors: dict[str, torch.Tensor], stats: dict[str, torch.Tensor]) -> dict[str, Any]:
        return {}

    def _generate_weights(self, strategy_output: dict[str, Any]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        device = next(iter(self.w_i)).device
        reg = torch.tensor(0.0, device=device)
        weights = {task: self.w_i[i] for i, task in enumerate(self.task_names)}
        return weights, reg

    def _assemble_loss(self, loss_tensors: dict[str, torch.Tensor], weights: dict[str, torch.Tensor], regularization: torch.Tensor) -> torch.Tensor:
        loss_list = [loss_tensors[task] for task in self.task_names]
        loss_stacked = torch.stack(loss_list)
        
        if self.step_count == 0:
            self.initial_losses = loss_stacked.detach().clone()
            
        if self.training and self.shared_param is not None and self.shared_param.requires_grad:
            w_i_param = self.w_i.clone().requires_grad_(True)
            
            norms = []
            for i in range(len(self.task_names)):
                grad = torch.autograd.grad(
                    loss_list[i], 
                    self.shared_param, 
                    retain_graph=True, 
                    allow_unused=True
                )[0]
                if grad is None:
                    norms.append(torch.tensor(0.0, device=self.w_i.device))
                else:
                    norms.append(torch.norm(w_i_param[i] * grad, 2))
                    
            norms = torch.stack(norms)
            mean_norm = norms.mean().detach()
            
            loss_ratios = loss_stacked.detach() / (self.initial_losses + 1e-8)
            inverse_training_rates = loss_ratios / loss_ratios.mean()
            target_norms = mean_norm * (inverse_training_rates ** self.alpha)
            
            grad_loss = torch.sum(torch.abs(norms - target_norms.detach()))
            
            w_grad = torch.autograd.grad(grad_loss, w_i_param)[0]
            
            with torch.no_grad():
                self.w_i -= self.lr * w_grad
                self.w_i.clamp_(min=1e-4)
                self.w_i.copy_((self.w_i / self.w_i.sum()) * len(self.task_names))
                
        self.step_count += 1
        
        total_loss = torch.tensor(0.0, device=loss_stacked.device)
        for i, task in enumerate(self.task_names):
            total_loss = total_loss + self.w_i[i].detach() * loss_tensors[task]
            
        return total_loss + regularization

    def _record_telemetry(self, weights: dict[str, torch.Tensor], strategy_output: dict[str, Any]) -> None:
        if self.recorder.record_enabled:
            trace = OptimizationReasoningTrace(
                optimization_type="GRADNORM",
                effective_task_weights={task: self.w_i[i].detach().clone() for i, task in enumerate(self.task_names)}
            )
            self.recorder.record(trace)
