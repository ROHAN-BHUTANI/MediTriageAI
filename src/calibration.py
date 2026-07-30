import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
import numpy as np

def expected_calibration_error(y_true, y_prob, num_bins=10):
    """Computes ECE across given probabilities."""
    y_pred = np.argmax(y_prob, axis=1)
    confidences = np.max(y_prob, axis=1)
    
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = np.zeros(1)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = (y_pred[in_bin] == y_true[in_bin]).mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return ece.item()

class TemperatureScaler(nn.Module):
    """
    Applies temperature scaling to calibrate logits.
    """
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        
    def forward(self, logits):
        return logits / self.temperature

class Calibrator:
    def __init__(self):
        self.specialist_scaler = TemperatureScaler()
        self.severity_scaler = TemperatureScaler()
        
    def fit(self, val_spec_logits: torch.Tensor, val_spec_labels: torch.Tensor,
                  val_sev_logits: torch.Tensor, val_sev_labels: torch.Tensor,
                  output_dir: str):
        """Fit the temperature using LBFGS on validation logits."""
        # 1. Filter out masked (-1) samples
        spec_mask = val_spec_labels != -1
        sev_mask = val_sev_labels != -1
        
        val_spec_logits = val_spec_logits[spec_mask]
        val_spec_labels = val_spec_labels[spec_mask]
        
        val_sev_logits = val_sev_logits[sev_mask]
        val_sev_labels = val_sev_labels[sev_mask]
        
        # Calculate before ECE
        before_spec_ece = expected_calibration_error(
            val_spec_labels.numpy(), 
            torch.softmax(val_spec_logits, dim=1).numpy()
        ) if len(val_spec_labels) > 0 else 0.0
        
        before_sev_ece = expected_calibration_error(
            val_sev_labels.numpy(), 
            torch.softmax(val_sev_logits, dim=1).numpy()
        ) if len(val_sev_labels) > 0 else 0.0
        
        # Fit Specialist Temperature
        if len(val_spec_labels) > 0:
            nll_criterion = nn.CrossEntropyLoss()
            optimizer = optim.LBFGS([self.specialist_scaler.temperature], lr=0.01, max_iter=50)
            
            def spec_eval():
                optimizer.zero_grad()
                loss = nll_criterion(self.specialist_scaler(val_spec_logits), val_spec_labels)
                loss.backward()
                return loss
            optimizer.step(spec_eval)
            
        # Fit Severity Temperature
        if len(val_sev_labels) > 0:
            nll_criterion = nn.CrossEntropyLoss()
            optimizer = optim.LBFGS([self.severity_scaler.temperature], lr=0.01, max_iter=50)
            
            def sev_eval():
                optimizer.zero_grad()
                loss = nll_criterion(self.severity_scaler(val_sev_logits), val_sev_labels)
                loss.backward()
                return loss
            optimizer.step(sev_eval)
            
        # Calculate after ECE
        after_spec_ece = expected_calibration_error(
            val_spec_labels.numpy(), 
            torch.softmax(self.specialist_scaler(val_spec_logits).detach(), dim=1).numpy()
        ) if len(val_spec_labels) > 0 else 0.0
        
        after_sev_ece = expected_calibration_error(
            val_sev_labels.numpy(), 
            torch.softmax(self.severity_scaler(val_sev_logits).detach(), dim=1).numpy()
        ) if len(val_sev_labels) > 0 else 0.0
        
        report = {
            "specialist": {
                "temperature": self.specialist_scaler.temperature.item(),
                "ece_before": before_spec_ece,
                "ece_after": after_spec_ece
            },
            "severity": {
                "temperature": self.severity_scaler.temperature.item(),
                "ece_before": before_sev_ece,
                "ece_after": after_sev_ece
            }
        }
        
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "calibration_report.json"), "w") as f:
            json.dump(report, f, indent=2)
            
        return report
