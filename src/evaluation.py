import os
import json
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from src.model import SPECIALIST_CLASSES, SEVERITY_LABELS
from src.config_manager import TrainingConfig

def compute_entropy(probs: np.ndarray) -> np.ndarray:
    """Computes Shannon entropy across the probability distribution."""
    # Add epsilon to prevent log(0)
    eps = 1e-10
    return -np.sum(probs * np.log(probs + eps), axis=-1)

class EvaluationExporter:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.predictions: List[Dict[str, Any]] = []
        
    def add_batch(
        self, 
        ids: List[str], 
        splits: List[str], 
        sources: List[str], 
        languages: List[str],
        spec_logits: torch.Tensor,
        sev_logits: torch.Tensor,
        spec_labels: torch.Tensor,
        sev_labels: torch.Tensor
    ):
        spec_probs = torch.softmax(spec_logits, dim=-1).detach().cpu().numpy()
        sev_probs = torch.softmax(sev_logits, dim=-1).detach().cpu().numpy()
        spec_preds = spec_probs.argmax(axis=-1)
        sev_preds = sev_probs.argmax(axis=-1)
        
        spec_labels_np = spec_labels.cpu().numpy()
        sev_labels_np = sev_labels.cpu().numpy()
        
        spec_entropy = compute_entropy(spec_probs)
        sev_entropy = compute_entropy(sev_probs)
        
        for i in range(len(ids)):
            s_label = spec_labels_np[i]
            s_pred = spec_preds[i]
            v_label = sev_labels_np[i]
            v_pred = sev_preds[i]
            
            s_conf = spec_probs[i][s_pred]
            v_conf = sev_probs[i][v_pred]
            
            gt_spec = SPECIALIST_CLASSES[s_label] if s_label != -1 else "UNKNOWN"
            pr_spec = SPECIALIST_CLASSES[s_pred]
            gt_sev = SEVERITY_LABELS[v_label] if v_label != -1 else "UNKNOWN"
            pr_sev = SEVERITY_LABELS[v_pred]
            
            self.predictions.append({
                "sample_id": ids[i],
                "split": splits[i],
                "dataset_source": sources[i],
                "language": languages[i],
                "ground_truth_department": gt_spec,
                "predicted_department": pr_spec,
                "department_confidence": float(s_conf),
                "department_entropy": float(spec_entropy[i]),
                "department_correct": bool(s_label == s_pred) if s_label != -1 else None,
                "ground_truth_severity": gt_sev,
                "predicted_severity": pr_sev,
                "severity_confidence": float(v_conf),
                "severity_entropy": float(sev_entropy[i]),
                "severity_correct": bool(v_label == v_pred) if v_label != -1 else None,
                "entropy": float(spec_entropy[i] + sev_entropy[i])
            })
            
    def export(self):
        df = pd.DataFrame(self.predictions)
        
        # 1. Predictions
        df.to_csv(os.path.join(self.output_dir, "predictions.csv"), index=False)
        df.to_parquet(os.path.join(self.output_dir, "predictions.parquet"), index=False)
        
        # 2. Misclassified vs Correct
        if "department_correct" in df.columns:
            mis_mask = (df["department_correct"] == False) | (df["severity_correct"] == False)
            cor_mask = (df["department_correct"] == True) & (df["severity_correct"] == True)
            
            df[mis_mask].to_csv(os.path.join(self.output_dir, "misclassified.csv"), index=False)
            df[cor_mask].to_csv(os.path.join(self.output_dir, "correct.csv"), index=False)
            
        # 3. Distributions
        if not df.empty:
            df[["department_confidence", "severity_confidence"]].describe().to_csv(
                os.path.join(self.output_dir, "confidence_distribution.csv")
            )
            df[["department_entropy", "severity_entropy", "entropy"]].describe().to_csv(
                os.path.join(self.output_dir, "entropy_distribution.csv")
            )

def generate_training_report(
    output_dir: str,
    config: TrainingConfig,
    experiment_id: str,
    git_commit: str,
    duration_seconds: float,
    primary_metric_val: float,
    dataset_manifest_hash: str = "N/A"
):
    import platform
    import time
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Training metadata
    metadata = {
        "experiment_id": experiment_id,
        "git_commit": git_commit,
        "dataset_manifest_hash": dataset_manifest_hash,
        "config_hash": "TODO_config_hash",
        "training_duration_seconds": duration_seconds,
        "primary_metric": primary_metric_val
    }
    with open(os.path.join(output_dir, "training_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
        
    # 2. Experiment manifest
    with open(os.path.join(output_dir, "experiment_manifest.json"), "w") as f:
        manifest = config.__dict__.copy()
        manifest.update(metadata)
        json.dump(manifest, f, indent=2)
        
    # 3. Hardware report
    hardware = {
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count(),
        "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__,
        "platform": platform.platform()
    }
    with open(os.path.join(output_dir, "hardware_report.json"), "w") as f:
        json.dump(hardware, f, indent=2)
        
    # 4. Evaluation summary template (written later or externally, just stub here if needed)
    with open(os.path.join(output_dir, "evaluation_summary.json"), "w") as f:
        json.dump({"status": "completed"}, f)
        
    # 5. Markdown summary
    md_content = f"""# Training Summary

* **Experiment ID:** {experiment_id}
* **Commit:** {git_commit}
* **Dataset Hash:** {dataset_manifest_hash}
* **Duration:** {duration_seconds:.2f} seconds
* **Primary Metric Value:** {primary_metric_val:.4f}

## Hardware
* **PyTorch Version:** {torch.__version__}
* **CUDA:** {torch.version.cuda}
* **GPUs:** {torch.cuda.device_count()}

## Configuration
```json
{json.dumps(config.__dict__, indent=2)}
```
"""
    with open(os.path.join(output_dir, "training_summary.md"), "w") as f:
        f.write(md_content)
