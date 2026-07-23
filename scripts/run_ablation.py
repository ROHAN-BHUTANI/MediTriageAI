"""Reproducible configuration-driven ablation study framework for E-PATH-CO-REASON."""

import os
import sys
import json
import time
import shutil
import argparse
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, XLMRobertaConfig

# Resolve repository root
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.emergent_path_triage.model import EmergentPathTriageModel, EmergentPathTriageConfig
from models.emergent_path_triage.types import RoutingDecision, ThoughtPath, EvidenceRepresentation
from src.data_pipeline import (
    get_leakage_safe_splits,
    TokenizerPipeline,
    EmergentTriageDataset,
    get_dataloader,
    LabelValidator,
)
from src.trainer import get_git_commit
from src.model import SPECIALIST_CLASSES, SEVERITY_LABELS, JointLoss

class AblationRegistry:
    """Registry managing modular, configuration-driven ablation experiments."""
    
    def __init__(self):
        self._registry = {}
        
    def register(self, name: str, config_updates: dict, description: str = ""):
        self._registry[name] = {
            "config_updates": config_updates,
            "description": description
        }
        
    def get_experiments(self) -> dict:
        return self._registry

def find_latest_checkpoint(results_dir="results") -> Path:
    results_path = Path(results_dir)
    checkpoint_files = list(results_path.glob("**/*.pt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files (.pt) found under: {results_dir}")
    return max(checkpoint_files, key=lambda p: p.stat().st_mtime)

def instantiate_fresh_model(checkpoint_path: Path, config_updates: dict, device: torch.device):
    """Instantiate a fresh model instance from checkpoint with explicit config updates."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_state = checkpoint.get("model_state_dict", checkpoint)
    
    # Reconstruct original transformer architecture parameters
    hidden_size = 768
    num_hidden_layers = 12
    vocab_size = 250002
    for k, v in model_state.items():
        if "word_embeddings.weight" in k:
            hidden_size = v.shape[1]
            vocab_size = v.shape[0]
            break
            
    layer_indices = set()
    for k in model_state.keys():
        if "encoder.layer." in k or "encoder.encoder.layer." in k:
            parts = k.split(".")
            for part in parts:
                if part.isdigit():
                    layer_indices.add(int(part))
                    break
    if layer_indices:
        num_hidden_layers = max(layer_indices) + 1
        
    latent_dim = 128
    for k, v in model_state.items():
        if "classifier_specialist.fc1.weight" in k:
            latent_dim = v.shape[1]
            break
            
    # Build E-PATH-CO-REASON configuration
    triage_config = EmergentPathTriageConfig(latent_dim=latent_dim)
    
    # Apply configuration-driven ablation overrides
    for key, val in config_updates.items():
        if hasattr(triage_config, key):
            setattr(triage_config, key, val)
        else:
            raise AttributeError(f"Invalid ablation configuration parameter: '{key}'")
            
    model_meta = EmergentPathTriageModel()
    model_config = XLMRobertaConfig(
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=2 if hidden_size < 100 else 12,
        intermediate_size=hidden_size * 2 if hidden_size < 100 else hidden_size * 4,
        max_position_embeddings=512,
        vocab_size=vocab_size
    )
    
    model = model_meta.build(model_config, triage_config=triage_config)
    model.load_state_dict(model_state, strict=False)
    model.to(device)
    model.eval()
    return model, triage_config

def count_active_parameters(model: torch.nn.Module, config: EmergentPathTriageConfig) -> int:
    """Compute count of active parameters based on the ablation configuration."""
    total = sum(p.numel() for p in model.parameters())
    inactive = 0
    
    if not getattr(config, "ablation_dces_enabled", True):
        inactive += sum(p.numel() for p in model.dces.parameters())
        
    if not getattr(config, "ablation_router_enabled", True):
        inactive += sum(p.numel() for p in model.router.parameters())
        
    if not getattr(config, "ablation_engine_enabled", True):
        for block in model.blocks:
            inactive += sum(p.numel() for p in block.parameters())
    else:
        if not getattr(config, "ablation_ctb1_enabled", True):
            inactive += sum(p.numel() for p in model.blocks[0].parameters())
        if not getattr(config, "ablation_ctb2_enabled", True):
            inactive += sum(p.numel() for p in model.blocks[1].parameters())
        if not getattr(config, "ablation_ctb3_enabled", True):
            inactive += sum(p.numel() for p in model.blocks[2].parameters())
        if not getattr(config, "ablation_ctb4_enabled", True):
            inactive += sum(p.numel() for p in model.blocks[3].parameters())
            
    return total - inactive

def evaluate_model(model: torch.nn.Module, loader, device: torch.device) -> dict:
    """Evaluate model performance metrics, inference latency, and memory footprints."""
    all_spec_preds = []
    all_spec_targets = []
    all_sev_preds = []
    all_sev_targets = []
    all_confidences = []
    all_entropies = []
    
    loss_fn = JointLoss()
    total_loss = 0.0
    num_batches = 0
    
    # Synchronize and measure inference latency
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        
    start_time = time.perf_counter()
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_spec = batch["labels_specialist"].to(device)
            labels_sev = batch["labels_severity"].to(device)
            
            outputs = model(input_ids, attention_mask)
            
            # Loss calculations
            loss_dict = loss_fn(
                outputs.specialist_logits,
                outputs.severity_logits,
                labels_spec,
                labels_sev
            )
            total_loss += float(loss_dict["joint_loss"].item())
            num_batches += 1
            
            # Specialist details
            spec_probs = torch.softmax(outputs.specialist_logits, dim=-1)
            conf, preds = torch.max(spec_probs, dim=-1)
            
            all_spec_preds.extend(preds.cpu().numpy().tolist())
            all_spec_targets.extend(labels_spec.cpu().numpy().tolist())
            all_confidences.extend(conf.cpu().numpy().tolist())
            
            eps = 1e-15
            entropy = -torch.sum(spec_probs * torch.log(spec_probs + eps), dim=-1)
            all_entropies.extend(entropy.cpu().numpy().tolist())
            
            # Severity details
            sev_preds = outputs.severity_logits.argmax(dim=-1)
            all_sev_preds.extend(sev_preds.cpu().numpy().tolist())
            all_sev_targets.extend(labels_sev.cpu().numpy().tolist())
            
    if device.type == "cuda":
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - start_time) * 1000.0 / len(all_spec_targets)
    
    # Metrics calculations
    y_true_spec = np.array(all_spec_targets)
    y_pred_spec = np.array(all_spec_preds)
    y_true_sev = np.array(all_sev_targets)
    y_pred_sev = np.array(all_sev_preds)
    
    spec_acc = np.mean(y_true_spec == y_pred_spec)
    sev_acc = np.mean(y_true_sev == y_pred_sev)
    
    num_classes = len(SPECIALIST_CLASSES)
    precision_list = []
    recall_list = []
    f1_list = []
    support_list = []
    
    for c_id in range(num_classes):
        y_true_bin = (y_true_spec == c_id)
        y_pred_bin = (y_pred_spec == c_id)
        
        tp = np.sum(y_true_bin & y_pred_bin)
        fp = np.sum((~y_true_bin) & y_pred_bin)
        fn = np.sum(y_true_bin & (~y_pred_bin))
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        
        precision_list.append(prec)
        recall_list.append(rec)
        f1_list.append(f1)
        support_list.append(np.sum(y_true_bin))
        
    macro_precision = np.mean(precision_list)
    macro_recall = np.mean(recall_list)
    macro_f1 = np.mean(f1_list)
    
    total_support = np.sum(support_list)
    weighted_f1 = np.sum(np.array(f1_list) * np.array(support_list)) / total_support if total_support > 0 else 0.0
    
    mcc = 0.0
    try:
        from sklearn.metrics import matthews_corrcoef
        mcc = float(matthews_corrcoef(y_true_spec, y_pred_spec))
    except Exception:
        pass
        
    gpu_mem = 0.0
    if device.type == "cuda":
        gpu_mem = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
        
    return {
        "spec_acc": float(spec_acc),
        "sev_acc": float(sev_acc),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "mcc": float(mcc),
        "loss": float(total_loss / num_batches) if num_batches > 0 else 0.0,
        "avg_confidence": float(np.mean(all_confidences)),
        "avg_entropy": float(np.mean(all_entropies)),
        "latency_ms": float(latency_ms),
        "gpu_mem_mb": float(gpu_mem)
    }

def run_ablation_campaign():
    parser = argparse.ArgumentParser(description="Run E-PATH-CO-REASON configuration-driven ablation study.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint. If None, resolves the latest.")
    parser.add_argument("--dataset", type=str, default="meditriage/data/processed/dataset.csv", help="Path to the dataset CSV.")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for evaluation split reproducibility.")
    parser.add_argument("--dry-run", action="store_true", help="Perform checks and setup registry without running evaluations.")
    args = parser.parse_args()

    print("=" * 60)
    print("STARTING REPRODUCIBLE ABLATION STUDY FRAMEWORK")
    print("=" * 60)

    # 1. Setup Random Seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 2. Resolve Checkpoint
    checkpoint_path = None
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        try:
            checkpoint_path = find_latest_checkpoint()
            print(f"Identified latest checkpoint: {checkpoint_path}")
        except Exception as e:
            print(f"Error resolving checkpoint: {e}")
            sys.exit(1)

    if not checkpoint_path.exists():
        print(f"Error: Checkpoint path does not exist: {checkpoint_path}")
        sys.exit(1)

    # 3. Load Test Split Dataloader
    dataset_csv = Path(args.dataset)
    if not dataset_csv.exists():
        print(f"Error: Dataset CSV file not found: {dataset_csv}")
        sys.exit(1)

    df = pd.read_csv(dataset_csv).dropna(subset=["text"])
    _, _, test_df = get_leakage_safe_splits(df, seed=args.seed, stratify=False)
    dataset_size = len(test_df)
    
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    pipeline = TokenizerPipeline(tokenizer, max_length=64)
    validator = LabelValidator()

    texts = test_df["text"].tolist()
    spec_ids = [validator.validate_specialist(str(c)) for c in test_df["department_code"]]
    sev_ids = [validator.validate_severity(str(l)) for l in test_df["severity_heuristic"]]

    test_dataset = EmergentTriageDataset(texts, spec_ids, sev_ids, pipeline)
    test_loader = get_dataloader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution Target Device: {device}")

    # 4. Registry Configuration setup
    registry = AblationRegistry()
    
    # Base Reference Run
    registry.register("Full_Model_Baseline", {}, "Baseline model with all architectural layers enabled.")
    
    # Major architectural parts
    registry.register("Ablate_DCES_Encoder", {"ablation_dces_enabled": False}, "Bypasses DCES multi-aspect projection layer.")
    registry.register("Ablate_Router_Uniform", {"ablation_router_enabled": False}, "Bypasses Gumbel-Softmax Router; outputs uniform routing paths.")
    registry.register("Ablate_Execution_Engine", {"ablation_engine_enabled": False}, "Bypasses execution engine loop; uses initial context h_0 directly.")
    
    # Thought Block components
    registry.register("Ablate_CTB1_Identity", {"ablation_ctb1_enabled": False}, "Replaces Clinical Thought Block 1 with identity mapping.")
    registry.register("Ablate_CTB2_Identity", {"ablation_ctb2_enabled": False}, "Replaces Clinical Thought Block 2 with identity mapping.")
    registry.register("Ablate_CTB3_Identity", {"ablation_ctb3_enabled": False}, "Replaces Clinical Thought Block 3 with identity mapping.")
    registry.register("Ablate_CTB4_Identity", {"ablation_ctb4_enabled": False}, "Replaces Clinical Thought Block 4 with identity mapping.")
    
    # Depth and Multi-step reasoning
    registry.register("Ablate_Multi_Step_Depth_1", {"ablation_multistep_enabled": False}, "Disables multi-step reasoning by clamping max path depth to 1.")
    registry.register("Ablate_Reasoning_Depth_2", {"max_path_depth": 2}, "Reduces max path depth to 2.")

    # 5. Baseline Verification Step
    print("\nExecuting baseline verification check...")
    std_model, std_config = instantiate_fresh_model(checkpoint_path, {}, device)
    std_metrics = evaluate_model(std_model, test_loader, device)

    base_updates = {
        "ablation_router_enabled": True,
        "ablation_dces_enabled": True,
        "ablation_engine_enabled": True,
        "ablation_ctb1_enabled": True,
        "ablation_ctb2_enabled": True,
        "ablation_ctb3_enabled": True,
        "ablation_ctb4_enabled": True,
        "ablation_multistep_enabled": True,
    }
    base_model, base_config = instantiate_fresh_model(checkpoint_path, base_updates, device)
    base_metrics = evaluate_model(base_model, test_loader, device)

    # Compare floating point tolerance
    mismatch = False
    for metric_name in ["spec_acc", "sev_acc", "loss", "avg_confidence", "avg_entropy"]:
        diff = abs(std_metrics[metric_name] - base_metrics[metric_name])
        if diff > 1e-6:
            print(f"CRITICAL ERROR: Standard metrics differ from baseline for '{metric_name}'!")
            print(f"  Standard: {std_metrics[metric_name]}")
            print(f"  Baseline: {base_metrics[metric_name]}")
            print(f"  Diff: {diff}")
            mismatch = True

    if mismatch:
        print("Ablation baseline verification FAILED! Aborting execution.")
        sys.exit(1)
        
    print("Baseline verification SUCCESSFUL! Standalone baseline metrics are identical.")

    if args.dry_run:
        print("\nDry-run mode active. Setup complete. Skipping evaluation run.")
        print("Registered Experiments:")
        for name, info in registry.get_experiments().items():
            print(f"  - {name}: {info['description']} (Overrides: {info['config_updates']})")
        print("=" * 60)
        sys.exit(0)

    # 6. Execute Ablation Campaigns
    print("\nStarting ablation evaluation campaign...")
    results_records = []
    baseline_metrics = std_metrics.copy()
    baseline_metrics["active_parameters"] = count_active_parameters(std_model, std_config)
    
    # Store standard baseline metrics as reference
    results_records.append({
        "Experiment_Name": "Full_Model_Baseline",
        "Specialist_Accuracy": baseline_metrics["spec_acc"],
        "Severity_Accuracy": baseline_metrics["sev_acc"],
        "Macro_Precision": baseline_metrics["macro_precision"],
        "Macro_Recall": baseline_metrics["macro_recall"],
        "Macro_F1": baseline_metrics["macro_f1"],
        "Weighted_F1": baseline_metrics["weighted_f1"],
        "MCC": baseline_metrics["mcc"],
        "Loss": baseline_metrics["loss"],
        "Average_Confidence": baseline_metrics["avg_confidence"],
        "Average_Entropy": baseline_metrics["avg_entropy"],
        "Inference_Latency_ms": baseline_metrics["latency_ms"],
        "GPU_Memory_MB": baseline_metrics["gpu_mem_mb"],
        "Active_Parameters": int(baseline_metrics["active_parameters"])
    })

    experiments = registry.get_experiments()
    for exp_name, exp_info in experiments.items():
        if exp_name == "Full_Model_Baseline":
            continue # Already evaluated
            
        print(f"Running experiment: {exp_name}...")
        updates = exp_info["config_updates"]
        
        # Instantiate fresh model
        exp_model, exp_config = instantiate_fresh_model(checkpoint_path, updates, device)
        active_params = count_active_parameters(exp_model, exp_config)
        
        # Run evaluations
        metrics = evaluate_model(exp_model, test_loader, device)
        
        results_records.append({
            "Experiment_Name": exp_name,
            "Specialist_Accuracy": metrics["spec_acc"],
            "Severity_Accuracy": metrics["sev_acc"],
            "Macro_Precision": metrics["macro_precision"],
            "Macro_Recall": metrics["macro_recall"],
            "Macro_F1": metrics["macro_f1"],
            "Weighted_F1": metrics["weighted_f1"],
            "MCC": metrics["mcc"],
            "Loss": metrics["loss"],
            "Average_Confidence": metrics["avg_confidence"],
            "Average_Entropy": metrics["avg_entropy"],
            "Inference_Latency_ms": metrics["latency_ms"],
            "GPU_Memory_MB": metrics["gpu_mem_mb"],
            "Active_Parameters": int(active_params)
        })

    # Save to timestamped directory
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path("results/ablation") / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nCreated timestamped ablation output directory: {out_dir}")

    # Export ablation_results.csv (deterministic sorting by name)
    results_df = pd.DataFrame(results_records).sort_values("Experiment_Name")
    results_df.to_csv(out_dir / "ablation_results.csv", index=False, encoding="utf-8")

    # Export ablation_results.json
    results_dict = results_df.to_dict(orient="records")
    with open(out_dir / "ablation_results.json", "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=4)

    # Export performance_comparison.csv (calculating Deltas vs baseline)
    comparison_data = []
    base_row = [r for r in results_records if r["Experiment_Name"] == "Full_Model_Baseline"][0]
    
    for r in results_records:
        comparison_data.append({
            "Experiment_Name": r["Experiment_Name"],
            "Specialist_Accuracy_Delta": float(r["Specialist_Accuracy"] - base_row["Specialist_Accuracy"]),
            "Severity_Accuracy_Delta": float(r["Severity_Accuracy"] - base_row["Severity_Accuracy"]),
            "Macro_F1_Delta": float(r["Macro_F1"] - base_row["Macro_F1"]),
            "Latency_Delta_ms": float(r["Inference_Latency_ms"] - base_row["Inference_Latency_ms"]),
            "Active_Parameters": int(r["Active_Parameters"])
        })
    comparison_df = pd.DataFrame(comparison_data).sort_values("Experiment_Name")
    comparison_df.to_csv(out_dir / "performance_comparison.csv", index=False, encoding="utf-8")

    # Export experiment_metadata.json
    metadata = {
        "checkpoint_path": str(checkpoint_path.resolve()),
        "git_commit": get_git_commit(),
        "dataset_size": dataset_size,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "random_seed": args.seed,
        "model_configuration": std_config.to_dict(),
        "registered_experiments": experiments
    }
    with open(out_dir / "experiment_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    # Export ablation_summary.md (markdown report)
    md_lines = [
        "# Ablation Study Summary Report",
        "",
        "## Overall Experiment Performance Metrics",
        "| Experiment Name | Specialist Accuracy | Severity Accuracy | Macro F1 | Loss | Latency (ms/sample) | Active Params |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    ]
    for _, row in results_df.iterrows():
        md_lines.append(
            f"| `{row['Experiment_Name']}` | {row['Specialist_Accuracy']:.2%} | {row['Severity_Accuracy']:.2%} | "
            f"{row['Macro_F1']:.4f} | {row['Loss']:.4f} | {row['Inference_Latency_ms']:.3f} | {int(row['Active_Parameters']):,} |"
        )
        
    md_lines.extend([
        "",
        "## Performance Comparison Deltas (vs. Full Model Baseline)",
        "| Experiment Name | Specialist Accuracy Delta | Severity Accuracy Delta | Macro F1 Delta | Latency Delta (ms) | Active Params |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |"
    ])
    for _, row in comparison_df.iterrows():
        md_lines.append(
            f"| `{row['Experiment_Name']}` | {row['Specialist_Accuracy_Delta']:.2%:+} | {row['Severity_Accuracy_Delta']:.2%:+} | "
            f"{row['Macro_F1_Delta']:.4f:+} | {row['Latency_Delta_ms']:.3f:+} | {int(row['Active_Parameters']):,} |"
        )

    summary_content = "\n".join(md_lines)
    with open(out_dir / "ablation_summary.md", "w", encoding="utf-8") as f:
        f.write(summary_content)

    # 7. Update "latest" link/copy
    latest_dir = Path("results/ablation/latest")
    if latest_dir.exists():
        if latest_dir.is_symlink():
            latest_dir.unlink()
        else:
            shutil.rmtree(latest_dir)
            
    latest_dir.mkdir(parents=True, exist_ok=True)
    for item in out_dir.iterdir():
        if item.is_file():
            shutil.copy2(item, latest_dir / item.name)

    print("=" * 60)
    print("ABLATION STUDY CAMPAIGN EXECUTION COMPLETE")
    print(f"Artifacts successfully written to: {out_dir}")
    print(f"Convenience copy updated at:       {latest_dir}")
    print("=" * 60)

if __name__ == "__main__":
    run_ablation_campaign()
