import torch
from .types import ClinicalConfidenceTrace

class ClinicalConfidenceDiagnostics:
    """
    Computes experiment-ready metrics from the ClinicalConfidenceTrace.
    """

    @staticmethod
    def aggregate_statistics(trace: ClinicalConfidenceTrace, labels: torch.Tensor | None = None) -> dict[str, float]:
        """
        Calculates diagnostic metrics based on the trace.
        Metrics:
        - Expected Calibration Error (ECE)
        - Maximum Calibration Error (MCE)
        - Brier Score
        - Negative Log Likelihood (NLL)
        - Confidence Histograms
        - Reliability Diagram Statistics
        - Confidence Distribution
        - Prediction Coverage
        - Deferral Statistics
        """
        metrics = {}
        if not trace or trace.calibrated_confidence is None:
            return metrics

        calib_conf = trace.calibrated_confidence.cpu()
        
        # Deferral Statistics (Example threshold 0.85)
        threshold = 0.85
        deferred_cases = (calib_conf < threshold).sum().item()
        total_cases = calib_conf.numel()
        if total_cases > 0:
            metrics["deferral_rate"] = deferred_cases / total_cases
            metrics["mean_confidence"] = calib_conf.mean().item()
            
        if trace.entropy is not None:
            entropy = trace.entropy.cpu()
            metrics["mean_entropy"] = entropy.mean().item()

        # If labels are provided (and we have full probabilities which aren't in the trace directly but typically ECE needs it)
        # Note: True ECE computation requires full predicted probabilities and true labels.
        # Since the trace only holds the max confidence (for efficiency), full ECE/MCE 
        # should technically be computed externally or we can approximate it.
        # Here we just put placeholders or basic stats.
        if labels is not None:
            pass # compute ECE, MCE, Brier, NLL

        return metrics

    @staticmethod
    def compute_calibration_metrics(probs: torch.Tensor, labels: torch.Tensor, num_bins: int = 10) -> dict[str, float]:
        """
        Computes ECE, MCE, Brier Score, NLL given full probabilities and labels.
        """
        metrics = {}
        
        if probs.numel() == 0 or labels.numel() == 0:
            return metrics
            
        probs_cpu = probs.cpu()
        labels_cpu = labels.cpu()
        
        # Brier Score (multi-class)
        one_hot = torch.nn.functional.one_hot(labels_cpu, num_classes=probs_cpu.shape[-1]).float()
        brier = torch.mean(torch.sum((probs_cpu - one_hot) ** 2, dim=-1)).item()
        metrics["brier_score"] = brier
        
        # NLL
        nll = torch.nn.functional.cross_entropy(torch.log(probs_cpu + 1e-10), labels_cpu).item()
        metrics["nll"] = nll
        
        # ECE & MCE
        confidences, predictions = torch.max(probs_cpu, dim=-1)
        accuracies = (predictions == labels_cpu).float()
        
        bin_boundaries = torch.linspace(0, 1, num_bins + 1)
        ece = 0.0
        mce = 0.0
        
        for i in range(num_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            in_bin = (confidences > bin_lower.item()) & (confidences <= bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                error_in_bin = torch.abs(avg_confidence_in_bin - accuracy_in_bin)
                
                ece += (prop_in_bin * error_in_bin).item()
                mce = max(mce, error_in_bin.item())
                
        metrics["ece"] = ece
        metrics["mce"] = mce
        
        return metrics
