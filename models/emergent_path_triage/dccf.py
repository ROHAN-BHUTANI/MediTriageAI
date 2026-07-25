import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Any

from .types import ClinicalConfidenceOutput, ClinicalConfidenceTrace, ConfidenceRecorder
from .config import EmergentPathTriageConfig


class BaseConfidenceEstimator(nn.Module, ABC):
    """
    Abstract interface for Clinical Confidence Framework (DCCF).
    Enforces a strict 7-stage confidence execution pipeline.
    """
    
    def __init__(self, config: EmergentPathTriageConfig, num_classes: int) -> None:
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        self.recorder = ConfidenceRecorder()

    def estimate(self, logits: torch.Tensor) -> ClinicalConfidenceOutput:
        """
        Executes the full DCCF confidence pipeline.
        Pipeline:
        1. Logit Collection
        2. Confidence Estimation
        3. Calibration
        4. Confidence Quantification
        5. Clinical Confidence Output
        6. Confidence Telemetry
        7. Diagnostics (Handled externally via recorder/diagnostics module)
        """
        # 1. Logit Collection
        valid_logits = self._collect_logits(logits)
        
        # 2. Confidence Estimation
        estimator_params, estimator_meta = self._estimate_parameters(valid_logits)
        
        # 3. Calibration
        calibrated_probs, calib_meta = self._calibrate(valid_logits, estimator_params)
        
        # 4. Confidence Quantification
        confidence_score, uncertainty_score = self._quantify_confidence(calibrated_probs)
        
        # 5. Clinical Confidence Output
        output = ClinicalConfidenceOutput(
            calibrated_probabilities=calibrated_probs,
            confidence_score=confidence_score,
            uncertainty_score=uncertainty_score,
            estimator_metadata=estimator_meta,
            calibration_metadata=calib_meta,
            future_annotations={}
        )
        
        # 6. Confidence Telemetry
        self._record_telemetry(valid_logits, output)
        
        return output
        
    def _collect_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if len(logits.shape) != 2:
            raise ValueError(f"Logits must be 2D, got {logits.shape}")
        if logits.shape[1] != self.num_classes:
            raise ValueError(f"Expected {self.num_classes} classes, got {logits.shape[1]}")
        return logits

    @abstractmethod
    def _estimate_parameters(self, logits: torch.Tensor) -> tuple[Any, dict[str, Any]]:
        pass

    @abstractmethod
    def _calibrate(self, logits: torch.Tensor, estimator_params: Any) -> tuple[torch.Tensor, dict[str, Any]]:
        pass

    def _quantify_confidence(self, calibrated_probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Standard: Max probability is confidence, entropy is uncertainty
        confidence_score = calibrated_probs.max(dim=-1).values
        entropy = -torch.sum(calibrated_probs * torch.log(calibrated_probs + 1e-10), dim=-1)
        uncertainty_score = entropy
        return confidence_score, uncertainty_score

    def _record_telemetry(self, raw_logits: torch.Tensor, output: ClinicalConfidenceOutput) -> None:
        if self.recorder.record_enabled:
            raw_probs = torch.softmax(raw_logits, dim=-1)
            raw_conf = raw_probs.max(dim=-1).values
            
            trace = ClinicalConfidenceTrace(
                raw_confidence=raw_conf.detach().clone(),
                calibrated_confidence=output.confidence_score.detach().clone(),
                uncertainty_evolution={"entropy": output.uncertainty_score.detach().clone()},
                entropy=output.uncertainty_score.detach().clone(),
                estimator_diagnostics=output.estimator_metadata,
                calibration_metadata=output.calibration_metadata,
                selective_prediction_metadata={}
            )
            self.recorder.record(trace)

    @abstractmethod
    def fit(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        """Fit calibration parameters on a validation set."""
        pass


class IdentityEstimator(BaseConfidenceEstimator):
    """Uncalibrated softmax baseline."""
    
    def _estimate_parameters(self, logits: torch.Tensor) -> tuple[Any, dict[str, Any]]:
        return None, {"estimator": "IDENTITY"}

    def _calibrate(self, logits: torch.Tensor, estimator_params: Any) -> tuple[torch.Tensor, dict[str, Any]]:
        probs = torch.softmax(logits, dim=-1)
        return probs, {"temperature": 1.0}

    def fit(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        pass


class TemperatureScalingEstimator(BaseConfidenceEstimator):
    """Standard single-scalar Temperature Scaling."""
    
    def __init__(self, config: EmergentPathTriageConfig, num_classes: int) -> None:
        super().__init__(config, num_classes)
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def _estimate_parameters(self, logits: torch.Tensor) -> tuple[Any, dict[str, Any]]:
        return self.temperature, {"estimator": "TEMPERATURE"}

    def _calibrate(self, logits: torch.Tensor, estimator_params: Any) -> tuple[torch.Tensor, dict[str, Any]]:
        temp = estimator_params
        # Avoid division by zero
        temp = torch.clamp(temp, min=1e-3)
        scaled_logits = logits / temp
        probs = torch.softmax(scaled_logits, dim=-1)
        return probs, {"temperature": temp.item()}

    def fit(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        """Fits temperature scaling parameter on the validation set using L-BFGS."""
        import torch.optim as optim
        import logging
        logger = logging.getLogger(__name__)
        
        # We optimize the parameter self.temperature
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        
        # Move inputs to parameter device
        device = self.temperature.device
        logits = logits.to(device)
        labels = labels.to(device)
        
        def eval_loss():
            optimizer.zero_grad()
            temp = torch.clamp(self.temperature, min=1e-3)
            loss = torch.nn.functional.cross_entropy(logits / temp, labels)
            loss.backward()
            return loss
            
        optimizer.step(eval_loss)
        logger.info(f"Fitted Temperature Scaling parameter: {self.temperature.item():.4f}")


class VectorScalingEstimator(BaseConfidenceEstimator):
    """Class-wise scaling and biasing."""
    
    def __init__(self, config: EmergentPathTriageConfig, num_classes: int) -> None:
        super().__init__(config, num_classes)
        self.W = nn.Parameter(torch.ones(num_classes))
        self.b = nn.Parameter(torch.zeros(num_classes))

    def _estimate_parameters(self, logits: torch.Tensor) -> tuple[Any, dict[str, Any]]:
        return (self.W, self.b), {"estimator": "VECTOR"}

    def _calibrate(self, logits: torch.Tensor, estimator_params: Any) -> tuple[torch.Tensor, dict[str, Any]]:
        W, b = estimator_params
        scaled_logits = logits * W + b
        probs = torch.softmax(scaled_logits, dim=-1)
        return probs, {"W_mean": W.mean().item(), "b_mean": b.mean().item()}

    def fit(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        """Fits Vector Scaling parameters W and b on the validation set."""
        import torch.optim as optim
        import logging
        logger = logging.getLogger(__name__)
        
        optimizer = optim.Adam([self.W, self.b], lr=0.01)
        device = self.W.device
        logits = logits.to(device)
        labels = labels.to(device)
        
        prev_loss = float('inf')
        for step in range(100):
            optimizer.zero_grad()
            scaled_logits = logits * self.W + self.b
            loss = torch.nn.functional.cross_entropy(scaled_logits, labels)
            loss.backward()
            optimizer.step()
            
            if abs(prev_loss - loss.item()) < 1e-5:
                break
            prev_loss = loss.item()
        logger.info(f"Fitted Vector Scaling parameters. Final Loss: {prev_loss:.4f}")


class DirichletEstimator(BaseConfidenceEstimator):
    """Dirichlet prior-based density scaling (simplified version)."""
    
    def __init__(self, config: EmergentPathTriageConfig, num_classes: int) -> None:
        super().__init__(config, num_classes)
        # Simplified: W matrix mapped from logits to Dir parameters
        self.W = nn.Parameter(torch.eye(num_classes))
        self.b = nn.Parameter(torch.zeros(num_classes))

    def _estimate_parameters(self, logits: torch.Tensor) -> tuple[Any, dict[str, Any]]:
        return (self.W, self.b), {"estimator": "DIRICHLET"}

    def _calibrate(self, logits: torch.Tensor, estimator_params: Any) -> tuple[torch.Tensor, dict[str, Any]]:
        W, b = estimator_params
        # Transformation mapping logits to dirichlet alpha parameters
        z = torch.matmul(logits, W) + b
        # Alpha must be > 0. using softplus
        alpha = torch.nn.functional.softplus(z) + 1e-10
        # Expected probability from Dirichlet is alpha / sum(alpha)
        alpha_sum = torch.sum(alpha, dim=-1, keepdim=True)
        probs = alpha / alpha_sum
        return probs, {"alpha_mean": alpha.mean().item()}

    def fit(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        """Fits Dirichlet parameters W and b on the validation set."""
        import torch.optim as optim
        import logging
        logger = logging.getLogger(__name__)
        
        optimizer = optim.Adam([self.W, self.b], lr=0.01)
        device = self.W.device
        logits = logits.to(device)
        labels = labels.to(device)
        
        prev_loss = float('inf')
        for step in range(100):
            optimizer.zero_grad()
            z = torch.matmul(logits, self.W) + self.b
            alpha = torch.nn.functional.softplus(z) + 1e-10
            alpha_sum = torch.sum(alpha, dim=-1, keepdim=True)
            probs = alpha / alpha_sum
            
            # Minimize negative log likelihood of the expected probabilities
            loss = torch.nn.functional.nll_loss(torch.log(probs + 1e-10), labels)
            loss.backward()
            optimizer.step()
            
            if abs(prev_loss - loss.item()) < 1e-5:
                break
            prev_loss = loss.item()
        logger.info(f"Fitted Dirichlet parameters. Final Loss: {prev_loss:.4f}")
