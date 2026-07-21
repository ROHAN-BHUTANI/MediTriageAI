"""Baseline execution script for E-PATH-CO-REASON."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import AutoTokenizer

# Insert project root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.emergent_path_triage.model import EmergentPathTriageConfig, EmergentPathTriageModel
from src.data_pipeline import (
    EmergentPathDataConfig,
    EmergentTriageDataset,
    TokenizerPipeline,
    detect_colab_environment,
    get_dataloader,
    get_leakage_safe_splits,
    set_global_seeds,
)
from src.trainer import EmergentTrainer, EmergentTrainerConfig, get_git_commit


def run_baseline_campaign(max_samples: int | None = 200) -> None:
    """Run E-PATH-CO-REASON baseline training, validate metrics, routing statistics, and generate plots."""
    print("Initializing baseline training campaign...")
    
    # 1. Configuration
    data_config = EmergentPathDataConfig()
    trainer_config = EmergentTrainerConfig(
        epochs=3,
        learning_rate=2e-4,
        encoder_lr=2e-5,
        weight_decay=0.01,
        gradient_clipping=1.0,
        gradient_accumulation_steps=1,
        use_amp=False,
        early_stopping_patience=2,
        checkpoint_dir="./results/baseline_campaign"
    )
    
    set_global_seeds(trainer_config.seed)
    
    # 2. Load and Split Dataset
    if not os.path.exists(data_config.dataset_path):
        raise FileNotFoundError(f"Processed dataset not found at {data_config.dataset_path}")
        
    df = pd.read_csv(data_config.dataset_path)
    if df["text"].isna().sum() > 0:
        df = df.dropna(subset=["text"])
        
    if max_samples is not None and len(df) > max_samples:
        print(f"Limiting dataset to {max_samples} samples for local campaign validation.")
        df = df.sample(max_samples, random_state=trainer_config.seed)

    train_df, val_df, test_df = get_leakage_safe_splits(
        df,
        train_ratio=data_config.train_ratio,
        val_ratio=data_config.val_ratio,
        test_ratio=data_config.test_ratio,
        seed=trainer_config.seed,
        stratify=False
    )
    
    # 3. Tokenization & DataLoaders
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    pipeline = TokenizerPipeline(tokenizer, max_length=data_config.max_length)
    
    # Mappings
    from src.data_pipeline import LabelValidator
    validator = LabelValidator()
    
    def process_df(target_df: pd.DataFrame) -> EmergentTriageDataset:
        texts = target_df["text"].tolist()
        spec_ids = [validator.validate_specialist(str(c)) for c in target_df["department_code"]]
        sev_ids = [validator.validate_severity(str(l)) for l in target_df["severity_heuristic"]]
        return EmergentTriageDataset(texts, spec_ids, sev_ids, pipeline)

    train_ds = process_df(train_df)
    val_ds = process_df(val_df)
    test_ds = process_df(test_df)
    
    train_loader = get_dataloader(train_ds, batch_size=data_config.batch_size, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=data_config.batch_size, shuffle=False)
    test_loader = get_dataloader(test_ds, batch_size=data_config.batch_size, shuffle=False)
    
    # 4. Model Building
    # Use a tiny configuration layer to save RAM/CPU computation
    class TinyConfig:
        hidden_size: int = 16
        num_hidden_layers: int = 1
        num_attention_heads: int = 2
        intermediate_size: int = 32
        max_position_embeddings: int = 32
        
    config = EmergentPathTriageConfig(latent_dim=8)
    model_meta = EmergentPathTriageModel()
    model = model_meta.build(TinyConfig(), triage_config=config)
    
    # Freeze the base encoder weights to speed up training locally
    for name, param in model.named_parameters():
        if "encoder" in name:
            param.requires_grad = False
            
    # 5. Trainer
    trainer = EmergentTrainer(
        model=model,
        config=trainer_config,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer
    )
    
    # Fit model
    best_val_metrics = trainer.fit()
    
    # Load best checkpoint for final evaluation
    best_ckpt_path = Path(trainer_config.checkpoint_dir) / "best_model.pt"
    trainer.load_checkpoint(best_ckpt_path)
    
    # 6. Evaluation on Test Split
    print("Evaluating baseline on Test Split...")
    model.eval()
    all_spec_labels = []
    all_sev_labels = []
    all_spec_preds = []
    all_sev_preds = []
    
    # Routing statistics lists
    all_routing_probs = []
    all_routing_argmax = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(trainer.device)
            attention_mask = batch["attention_mask"].to(trainer.device)
            labels_spec = batch["labels_specialist"].to(trainer.device)
            labels_sev = batch["labels_severity"].to(trainer.device)
            
            outputs = model(input_ids, attention_mask)
            
            # Predict
            spec_preds = outputs.specialist_logits.argmax(dim=-1).cpu().numpy()
            sev_preds = outputs.severity_logits.argmax(dim=-1).cpu().numpy()
            
            all_spec_labels.extend(labels_spec.cpu().numpy())
            all_sev_labels.extend(labels_sev.cpu().numpy())
            all_spec_preds.extend(spec_preds)
            all_sev_preds.extend(sev_preds)
            
            # Collect last routing decision
            # DCRR outputs RoutingDecision with shape [B, M, N]
            decisions = model._last_routing_decision
            if decisions is not None:
                probs = decisions.routing_probabilities.cpu().numpy() # [B, M, N]
                all_routing_probs.append(probs)
                all_routing_argmax.append(probs.argmax(axis=-1))

    # Calculate metrics
    spec_acc = accuracy_score(all_spec_labels, all_spec_preds)
    spec_p, spec_r, spec_f1, _ = precision_recall_fscore_support(
        all_spec_labels, all_spec_preds, average="macro", zero_division=0
    )
    
    sev_acc = accuracy_score(all_sev_labels, all_sev_preds)
    sev_p, sev_r, sev_f1, _ = precision_recall_fscore_support(
        all_sev_labels, all_sev_preds, average="macro", zero_division=0
    )

    print("\n================== BASELINE CAMPAIGN RESULTS ==================")
    print(f"Specialist Accuracy  : {spec_acc:.4f} | F1: {spec_f1:.4f}")
    print(f"Severity Accuracy    : {sev_acc:.4f}  | F1: {sev_f1:.4f}")
    
    # 7. Save Metrics JSON
    metrics_json = {
        "reproducibility": {
            "random_seed": trainer_config.seed,
            "gpu_model": trainer.env_meta["gpu_name"],
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
            "pytorch_version": torch.__version__,
            "dataset_samples": len(df),
            "git_commit": get_git_commit(),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "specialist": {
            "accuracy": float(spec_acc),
            "macro_precision": float(spec_p),
            "macro_recall": float(spec_r),
            "macro_f1": float(spec_f1)
        },
        "severity": {
            "accuracy": float(sev_acc),
            "macro_precision": float(sev_p),
            "macro_recall": float(sev_r),
            "macro_f1": float(sev_f1)
        },
        "overall_losses": {
            "total_loss": float(best_val_metrics["val_loss"]),
            "specialist_loss": float(best_val_metrics["val_specialist_loss"]),
            "severity_loss": float(best_val_metrics["val_severity_loss"]),
            "consistency_loss": float(best_val_metrics["val_cons_loss"]),
            "diversity_loss": float(best_val_metrics["val_div_loss"]),
            "ortho_loss": float(best_val_metrics["val_ortho_loss"])
        }
    }
    
    out_dir = Path(trainer_config.checkpoint_dir)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_json, f, indent=4)

    # 8. Routing Analysis
    routing_stats = {}
    if all_routing_probs:
        all_probs = np.concatenate(all_routing_probs, axis=0) # [B, M, N]
        all_argmax = np.concatenate(all_routing_argmax, axis=0) # [B, M]
        B, M, N = all_probs.shape
        
        # Calculate Entropy
        epsilon = 1e-9
        entropies = -np.sum(all_probs * np.log(all_probs + epsilon), axis=-1) # [B, M]
        avg_entropy_per_step = entropies.mean(axis=0).tolist()
        mean_entropy = float(entropies.mean())
        
        # Confidence
        confidences = np.max(all_probs, axis=-1) # [B, M]
        avg_confidence_per_step = confidences.mean(axis=0).tolist()
        mean_confidence = float(confidences.mean())
        
        # CTB Frequencies per step
        utilization_freqs = []
        for step in range(M):
            counts = np.bincount(all_argmax[:, step], minlength=N)
            freqs = (counts / B).tolist()
            utilization_freqs.append(freqs)
            
        routing_stats = {
            "routing_entropy": {
                "mean_entropy": mean_entropy,
                "entropy_per_step": avg_entropy_per_step
            },
            "ctb_utilization_frequencies": utilization_freqs,
            "average_reasoning_depth": M,
            "routing_confidence": {
                "mean_confidence": mean_confidence,
                "confidence_per_step": avg_confidence_per_step
            }
        }
        
        with open(out_dir / "routing_statistics.json", "w", encoding="utf-8") as f:
            json.dump(routing_stats, f, indent=4)

    # 9. Confusion Matrices
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        confusion_matrix(all_spec_labels, all_spec_preds),
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=validator.specialist_classes,
        yticklabels=validator.specialist_classes
    )
    plt.title("Specialist Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(out_dir / "specialist_confusion_matrix.png")
    plt.close()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        confusion_matrix(all_sev_labels, all_sev_preds),
        annot=True,
        fmt="d",
        cmap="Oranges",
        xticklabels=validator.severity_labels,
        yticklabels=validator.severity_labels
    )
    plt.title("Severity Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(out_dir / "severity_confusion_matrix.png")
    plt.close()

    # 10. Plots
    history_df = pd.DataFrame(trainer.history)
    
    # Save validation_history.csv
    val_cols = [c for c in history_df.columns if "val_" in c or c in ["epoch", "time"]]
    history_df[val_cols].to_csv(out_dir / "validation_history.csv", index=False)
    
    # Loss curves plot
    plt.figure(figsize=(10, 6))
    plt.plot(history_df["epoch"], history_df["train_loss"], label="Train Loss", marker="o")
    plt.plot(history_df["epoch"], history_df["val_loss"], label="Val Loss", marker="x")
    plt.title("E-PATH-CO-REASON Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(out_dir / "loss_curves.png")
    plt.close()
    
    # Accuracy curves plot
    plt.figure(figsize=(10, 6))
    plt.plot(history_df["epoch"], history_df["train_specialist_acc"], label="Train Specialist Acc", marker="o")
    plt.plot(history_df["epoch"], history_df["val_specialist_acc"], label="Val Specialist Acc", marker="x")
    plt.plot(history_df["epoch"], history_df["train_severity_acc"], label="Train Severity Acc", marker="s")
    plt.plot(history_df["epoch"], history_df["val_severity_acc"], label="Val Severity Acc", marker="d")
    plt.title("E-PATH-CO-REASON Accuracy Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(out_dir / "accuracy_curves.png")
    plt.close()
    
    print(f"Baseline campaign files successfully exported to {out_dir}.")


if __name__ == "__main__":
    run_baseline_campaign()
