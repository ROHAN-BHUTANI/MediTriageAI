"""Prediction Error Analysis Framework for E-PATH-CO-REASON."""

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
# Resolve repository root
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_pipeline import (
    get_leakage_safe_splits,
    TokenizerPipeline,
    EmergentTriageDataset,
    get_dataloader,
    LabelValidator,
)
from src.checkpoint_manager import load_checkpoint, reconstruct_model_and_tokenizer
from src.trainer import get_git_commit
from src.model import SPECIALIST_CLASSES, SEVERITY_LABELS

def find_latest_checkpoint(results_dir="results") -> Path:
    results_path = Path(results_dir)
    checkpoint_files = list(results_path.glob("**/*.pt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files (.pt) found under: {results_dir}")
    # Return the one with the latest modification time
    latest_ckpt = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
    return latest_ckpt

def compute_entropy(probs: np.ndarray) -> float:
    # probs shape: (C,)
    return float(-np.sum(probs * np.log(probs + 1e-15)))

def run_analysis():
    parser = argparse.ArgumentParser(description="Run Prediction Error Analysis on the test set.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint. If None, automatically find the latest.")
    parser.add_argument("--dataset", type=str, default="data/processed/enriched/dataset_enriched.csv", help="Path to the dataset CSV.")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for split reproducibility.")
    args = parser.parse_args()

    print("=" * 60)
    print("STARTING PREDICTION ERROR ANALYSIS")
    print("=" * 60)

    # 1. Resolve Checkpoint
    checkpoint_path = None
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        try:
            checkpoint_path = find_latest_checkpoint()
            print(f"Automatically identified latest checkpoint: {checkpoint_path}")
        except Exception as e:
            print(f"Error finding checkpoint: {e}")
            sys.exit(1)

    if not checkpoint_path.exists():
        print(f"Error: Checkpoint path does not exist: {checkpoint_path}")
        sys.exit(1)

    # Load checkpoint and reconstruct model & tokenizer
    print("Loading checkpoint...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_data = load_checkpoint(checkpoint_path, map_location=device)
    model, tokenizer, model_meta = reconstruct_model_and_tokenizer(checkpoint_data, device=device)
    model.eval()

    # 3. Load Test Split Data
    dataset_csv = Path(args.dataset)
    if not dataset_csv.exists():
        print(f"Error: Dataset file not found at: {dataset_csv}")
        sys.exit(1)

    print(f"Loading dataset from: {dataset_csv}")
    df = pd.read_csv(dataset_csv)
    df = df.dropna(subset=["text"])

    # Keep a copy of test DataFrame for original complaints lookup
    # Split using the leakage safe splits
    _, _, test_df = get_leakage_safe_splits(df, seed=args.seed, stratify=False)
    dataset_size = len(test_df)
    print(f"Complete test split size: {dataset_size} samples")

    if dataset_size == 0:
        print("Error: Test split is empty!")
        sys.exit(1)

    pipeline = TokenizerPipeline(tokenizer, max_length=64)
    validator = LabelValidator()

    texts = test_df["text"].tolist()
    spec_ids = [validator.validate_specialist(str(c)) for c in test_df["department_code"]]
    sev_ids = [validator.validate_severity(str(l)) for l in test_df["severity_heuristic"]]

    test_dataset = EmergentTriageDataset(texts, spec_ids, sev_ids, pipeline)
    test_loader = get_dataloader(test_dataset, batch_size=32, shuffle=False)

    # 4. Run Inference and collect predictions
    print("Running inference over test set...")
    sample_records = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_spec = batch["labels_specialist"].cpu().numpy()
            labels_sev = batch["labels_severity"].cpu().numpy()
            
            outputs = model(input_ids, attention_mask)
            
            # Specialist Softmax & Predictions
            spec_probs = torch.softmax(outputs.specialist_logits, dim=-1).cpu().numpy()
            spec_preds = outputs.specialist_logits.argmax(dim=-1).cpu().numpy()
            
            # Severity Softmax & Predictions
            sev_probs = torch.softmax(outputs.severity_logits, dim=-1).cpu().numpy()
            sev_preds = outputs.severity_logits.argmax(dim=-1).cpu().numpy()
            
            # Routing Details
            routing_decision = getattr(model, "_last_routing_decision", None)
            
            # Process each sample in the batch
            batch_size = input_ids.shape[0]
            for i in range(batch_size):
                global_idx = batch_idx * 32 + i
                
                # Fetch original text and length
                orig_text = texts[global_idx]
                tokens = tokenizer.encode(orig_text)
                token_len = len(tokens)
                
                # True vs Predicted Specialists
                t_spec_id = int(labels_spec[i])
                t_spec_name = SPECIALIST_CLASSES[t_spec_id]
                p_spec_id = int(spec_preds[i])
                p_spec_name = SPECIALIST_CLASSES[p_spec_id]
                
                # Top-5 predicted specialists
                top5_ids = list(np.argsort(spec_probs[i])[::-1][:5])
                top5_names = [SPECIALIST_CLASSES[idx] for idx in top5_ids]
                top5_probs = [float(spec_probs[i][idx]) for idx in top5_ids]
                
                # True vs Predicted Severity
                t_sev_id = int(labels_sev[i])
                t_sev_name = SEVERITY_LABELS[t_sev_id]
                p_sev_id = int(sev_preds[i])
                p_sev_name = SEVERITY_LABELS[p_sev_id]
                
                # Correctness flags
                correct_spec = bool(t_spec_id == p_spec_id)
                correct_sev = bool(t_sev_id == p_sev_id)
                correct_joint = bool(correct_spec and correct_sev)
                
                # Entropy
                entropy_val = compute_entropy(spec_probs[i])
                
                # Routing path and selections
                reasoning_path_str = "[]"
                ctb_selections_str = "[]"
                r_confidence = 0.0
                
                if routing_decision is not None:
                    sel_blocks = routing_decision.selected_blocks
                    r_confidence = float(routing_decision.routing_confidence.item())
                    
                    reasoning_path_str = str(sel_blocks)
                    ctb_selections_str = str(sel_blocks)
                
                sample_records.append({
                    "sample_index": global_idx,
                    "original_complaint": orig_text,
                    "token_length": token_len,
                    "true_specialist_id": t_spec_id,
                    "true_specialist_name": t_spec_name,
                    "predicted_specialist_id": p_spec_id,
                    "predicted_specialist_name": p_spec_name,
                    "top5_predicted_specialist_ids": top5_ids,
                    "top5_predicted_specialist_names": top5_names,
                    "top5_probabilities": top5_probs,
                    "confidence": float(spec_probs[i][p_spec_id]),
                    "true_severity_id": t_sev_id,
                    "true_severity_name": t_sev_name,
                    "predicted_severity_id": p_sev_id,
                    "predicted_severity_name": p_sev_name,
                    "correctness_specialist": correct_spec,
                    "correctness_severity": correct_sev,
                    "correctness_joint": correct_joint,
                    "entropy": entropy_val,
                    "reasoning_path": reasoning_path_str,
                    "CTB_selections": ctb_selections_str,
                    "routing_confidence": r_confidence
                })

    # Save to Timestamped outputs directory
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path("results/prediction_error_analysis") / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created timestamped analysis output directory: {out_dir}")

    # Build DataFrame
    records_df = pd.DataFrame(sample_records)
    
    # 5. Compute metrics & save outputs with strict UTF-8
    
    # Correct and Misclassified outputs
    misclassified_df = records_df[~records_df["correctness_joint"]].sort_values("sample_index")
    correct_df = records_df[records_df["correctness_joint"]].sort_values("sample_index")
    
    misclassified_df.to_csv(out_dir / "misclassified_samples.csv", index=False, encoding="utf-8")
    correct_df.to_csv(out_dir / "correct_predictions.csv", index=False, encoding="utf-8")

    # High and Low confidence errors
    spec_errors_df = records_df[~records_df["correctness_specialist"]]
    
    # Sorting deterministically
    top100_high_conf = spec_errors_df.sort_values(by=["confidence", "sample_index"], ascending=[False, True]).head(100)
    top100_low_conf = spec_errors_df.sort_values(by=["confidence", "sample_index"], ascending=[True, True]).head(100)
    
    top100_high_conf.to_csv(out_dir / "top100_high_confidence_errors.csv", index=False, encoding="utf-8")
    top100_low_conf.to_csv(out_dir / "top100_low_confidence_errors.csv", index=False, encoding="utf-8")

    # Confusion Matrix
    y_true_spec = records_df["true_specialist_name"].tolist()
    y_pred_spec = records_df["predicted_specialist_name"].tolist()
    
    classes = sorted(SPECIALIST_CLASSES)
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    
    for t_val, p_val in zip(y_true_spec, y_pred_spec):
        cm[class_to_idx[t_val]][class_to_idx[p_val]] += 1
        
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    cm_df.index.name = "True_Class"
    cm_df.to_csv(out_dir / "confusion_matrix.csv", encoding="utf-8")
    
    # Normalized Confusion Matrix
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sums > 0, cm / row_sums, 0.0)
    cm_norm_df = pd.DataFrame(cm_norm, index=classes, columns=classes)
    cm_norm_df.index.name = "True_Class"
    cm_norm_df.to_csv(out_dir / "normalized_confusion_matrix.csv", encoding="utf-8")

    # Binned Distributions
    pred_dist_data = []
    total_samples = len(records_df)
    for c in classes:
        cnt = int((records_df["predicted_specialist_name"] == c).sum())
        pred_dist_data.append({
            "Class": c,
            "Count": cnt,
            "Percentage": float(cnt / total_samples) if total_samples > 0 else 0.0
        })
    pred_dist_df = pd.DataFrame(pred_dist_data).sort_values("Class")
    pred_dist_df.to_csv(out_dir / "prediction_distribution.csv", index=False, encoding="utf-8")

    # Bins for Confidence Distribution (0.0 to 1.0, 10 bins)
    conf_bins = np.linspace(0.0, 1.0, 11)
    conf_counts, _ = np.histogram(records_df["confidence"], bins=conf_bins)
    conf_labels = [f"[{conf_bins[j]:.1f}, {conf_bins[j+1]:.1f})" for j in range(9)] + [f"[{conf_bins[9]:.1f}, {conf_bins[10]:.1f}]"]
    conf_dist_df = pd.DataFrame({
        "Confidence Range": conf_labels,
        "Count": [int(x) for x in conf_counts],
        "Percentage": [float(x / total_samples) for x in conf_counts]
    })
    conf_dist_df.to_csv(out_dir / "confidence_distribution.csv", index=False, encoding="utf-8")

    # Bins for Entropy Distribution (0.0 to 2.6, 13 bins)
    ent_bins = np.linspace(0.0, 2.6, 14)
    ent_counts, _ = np.histogram(records_df["entropy"], bins=ent_bins)
    ent_labels = [f"[{ent_bins[j]:.1f}, {ent_bins[j+1]:.1f})" for j in range(12)] + [f"[{ent_bins[12]:.1f}, {ent_bins[13]:.1f}]"]
    ent_dist_df = pd.DataFrame({
        "Entropy Range": ent_labels,
        "Count": [int(x) for x in ent_counts],
        "Percentage": [float(x / total_samples) for x in ent_counts]
    })
    ent_dist_df.to_csv(out_dir / "entropy_distribution.csv", index=False, encoding="utf-8")

    # Specificity, Balanced Accuracy, MCC per Specialist
    per_class_data = []
    y_true_spec_ids = records_df["true_specialist_id"].to_numpy()
    y_pred_spec_ids = records_df["predicted_specialist_id"].to_numpy()

    for c in classes:
        c_id = validator.spec_to_id[c]
        y_true_bin = (y_true_spec_ids == c_id)
        y_pred_bin = (y_pred_spec_ids == c_id)
        
        tp = int(np.sum(y_true_bin & y_pred_bin))
        tn = int(np.sum((~y_true_bin) & (~y_pred_bin)))
        fp = int(np.sum((~y_true_bin) & y_pred_bin))
        fn = int(np.sum(y_true_bin & (~y_pred_bin)))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support = tp + fn
        
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        balanced_accuracy = (recall + specificity) / 2.0
        
        mcc_denom = np.sqrt(float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn))
        mcc = (tp * tn - fp * fn) / mcc_denom if mcc_denom > 0 else 0.0
        
        per_class_data.append({
            "Class": c,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Support": support,
            "Specificity": specificity,
            "Balanced Accuracy": balanced_accuracy,
            "Matthews Correlation Coefficient": mcc
        })
        
    per_class_df = pd.DataFrame(per_class_data).sort_values("Class")
    per_class_df.to_csv(out_dir / "per_class_metrics.csv", index=False, encoding="utf-8")

    # Classification Report JSON
    macro_precision = float(per_class_df["Precision"].mean())
    macro_recall = float(per_class_df["Recall"].mean())
    macro_f1 = float(per_class_df["F1"].mean())
    
    correct_spec_count = int(records_df["correctness_specialist"].sum())
    overall_accuracy = float(correct_spec_count / total_samples) if total_samples > 0 else 0.0
    
    total_support = float(per_class_df["Support"].sum())
    weighted_precision = float((per_class_df["Precision"] * per_class_df["Support"]).sum() / total_support) if total_support > 0 else 0.0
    weighted_recall = float((per_class_df["Recall"] * per_class_df["Support"]).sum() / total_support) if total_support > 0 else 0.0
    weighted_f1 = float((per_class_df["F1"] * per_class_df["Support"]).sum() / total_support) if total_support > 0 else 0.0
    
    report_dict = {
        "accuracy": overall_accuracy,
        "macro avg": {
            "precision": macro_precision,
            "recall": macro_recall,
            "f1-score": macro_f1,
            "support": int(total_support)
        },
        "weighted avg": {
            "precision": weighted_precision,
            "recall": weighted_recall,
            "f1-score": weighted_f1,
            "support": int(total_support)
        }
    }
    for row in per_class_data:
        report_dict[row["Class"]] = {
            "precision": row["Precision"],
            "recall": row["Recall"],
            "f1-score": row["F1"],
            "support": int(row["Support"])
        }
        
    with open(out_dir / "classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=4)

    # 6. Generate prediction_error_summary.md
    confusions = {}
    for t_val, p_val in zip(y_true_spec, y_pred_spec):
        if t_val != p_val:
            pair = (t_val, p_val)
            confusions[pair] = confusions.get(pair, 0) + 1
            
    sorted_confusions = sorted(confusions.items(), key=lambda x: x[1], reverse=True)[:5]
    sorted_f1 = per_class_df.sort_values("F1").head(5)
    high_conf_errors = top100_high_conf.head(5)
    
    correct_spec_only_df = records_df[records_df["correctness_specialist"]]
    low_conf_correct = correct_spec_only_df.sort_values(by=["confidence", "sample_index"], ascending=[True, True]).head(5)

    md_lines = [
        "# Prediction Error Analysis Summary Report",
        "",
        "## Overall Diagnostic Performance",
        f"- **Total Test Samples**: {total_samples}",
        f"- **Specialist Accuracy**: {overall_accuracy:.2%}",
        f"- **Macro-averaged Specialist F1-score**: {macro_f1:.4f}",
        f"- **Weighted-averaged Specialist F1-score**: {weighted_f1:.4f}",
        "",
        "## Most Confused Specialist Pairs",
        "The following true -> predicted class pairings represent the most common errors made by the model:",
        "| True Class | Predicted Class | Count |",
        "| :--- | :--- | :--- |"
    ]
    for (t_c, p_c), count in sorted_confusions:
        md_lines.append(f"| `{t_c}` | `{p_c}` | {count} |")
        
    md_lines.extend([
        "",
        "## Most Difficult Specialists",
        "Specialists with the lowest F1-score performance:",
        "| Specialist Class | Precision | Recall | F1 Score | Support |",
        "| :--- | :--- | :--- | :--- | :--- |"
    ])
    for _, row in sorted_f1.iterrows():
        md_lines.append(f"| `{row['Class']}` | {row['Precision']:.4f} | {row['Recall']:.4f} | {row['F1']:.4f} | {row['Support']} |")

    md_lines.extend([
        "",
        "## Highest Confidence Wrong Predictions",
        "Errors where the model made incorrect predictions with extremely high confidence:",
        "| Index | True Class | Predicted Class | Confidence | Complaint |",
        "| :--- | :--- | :--- | :--- | :--- |"
    ])
    for _, row in high_conf_errors.iterrows():
        complaint_trunc = row['original_complaint'][:100] + "..." if len(row['original_complaint']) > 100 else row['original_complaint']
        md_lines.append(f"| {row['sample_index']} | `{row['true_specialist_name']}` | `{row['predicted_specialist_name']}` | {row['confidence']:.2%} | {complaint_trunc} |")

    md_lines.extend([
        "",
        "## Lowest Confidence Correct Predictions",
        "Correct predictions where the model had the lowest confidence score:",
        "| Index | Specialist Class | Confidence | Complaint |",
        "| :--- | :--- | :--- | :--- |"
    ])
    for _, row in low_conf_correct.iterrows():
        complaint_trunc = row['original_complaint'][:100] + "..." if len(row['original_complaint']) > 100 else row['original_complaint']
        md_lines.append(f"| {row['sample_index']} | `{row['true_specialist_name']}` | {row['confidence']:.2%} | {complaint_trunc} |")

    md_lines.extend([
        "",
        "## Class-wise Failure Analysis",
        "Detailed performance per specialist class:",
        "| Specialist Class | Precision | Recall | F1 | Specificity | Balanced Accuracy | MCC | Support |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    ])
    for _, row in per_class_df.iterrows():
        md_lines.append(
            f"| `{row['Class']}` | {row['Precision']:.4f} | {row['Recall']:.4f} | {row['F1']:.4f} | "
            f"{row['Specificity']:.4f} | {row['Balanced Accuracy']:.4f} | "
            f"{row['Matthews Correlation Coefficient']:.4f} | {row['Support']} |"
        )

    summary_content = "\n".join(md_lines)
    with open(out_dir / "prediction_error_summary.md", "w", encoding="utf-8") as f:
        f.write(summary_content)

    # 7. Write metadata.json
    metadata = {
        "checkpoint_path": str(checkpoint_path.resolve()),
        "git_commit": get_git_commit(),
        "dataset_size": dataset_size,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "random_seed": args.seed,
        "model_configuration": model.config.to_dict() if (hasattr(model, "config") and hasattr(model.config, "to_dict")) else checkpoint_data.get("config", {})
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    # 8. Create or update the "latest" link/copy
    latest_dir = Path("results/prediction_error_analysis/latest")
    if latest_dir.exists():
        if latest_dir.is_symlink():
            latest_dir.unlink()
        else:
            shutil.rmtree(latest_dir)
            
    # Copy directory contents to "latest" to avoid Windows symlink permission limits
    latest_dir.mkdir(parents=True, exist_ok=True)
    for item in out_dir.iterdir():
        if item.is_file():
            shutil.copy2(item, latest_dir / item.name)

    print("=" * 60)
    print("ANALYSIS EXECUTION COMPLETE")
    print(f"Artifacts successfully written to: {out_dir}")
    print(f"Convenience copy updated at:       {latest_dir}")
    print(f"Overall Accuracy:                  {overall_accuracy:.2%}")
    print(f"Overall Macro F1:                  {macro_f1:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    run_analysis()
