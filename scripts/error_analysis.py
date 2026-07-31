import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path

def main():
    results_dir = Path("results/error_analysis")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for latest predictions
    pred_path = Path("results/emergent_path_triage/predictions.csv")
    if not pred_path.exists():
        print("predictions.csv not found!")
        return

    df = pd.read_csv(pred_path)
    
    # 1. Per-class Precision, 2. Recall, 3. F1, 4. Support
    y_true = df["ground_truth_department"].fillna("UNKNOWN").astype(str)
    y_pred = df["predicted_department"].fillna("UNKNOWN").astype(str)
    
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(results_dir / "classification_report.csv")
    
    classes = sorted(list(set(y_true) | set(y_pred)))
    
    # 5. Confusion Matrix, 6. Normalized Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    cm_df.to_csv(results_dir / "confusion_matrix.csv")
    
    cm_norm = confusion_matrix(y_true, y_pred, labels=classes, normalize="true")
    cm_norm_df = pd.DataFrame(cm_norm, index=classes, columns=classes)
    cm_norm_df.to_csv(results_dir / "confusion_matrix_normalized.csv")
    
    # Plots
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_norm_df, annot=False, cmap="Blues")
    plt.title("Normalized Confusion Matrix")
    plt.tight_layout()
    plt.savefig(results_dir / "confusion_matrix.png")
    plt.close()
    
    # 7. Prediction Distribution, 8. Ground Truth Distribution
    pred_dist = y_pred.value_counts()
    gt_dist = y_true.value_counts()
    
    dist_df = pd.DataFrame({"Ground_Truth": gt_dist, "Predicted": pred_dist}).fillna(0)
    dist_df.to_csv(results_dir / "class_distributions.csv")
    
    dist_df.plot(kind="bar", figsize=(12, 6))
    plt.title("Class Distributions")
    plt.tight_layout()
    plt.savefig(results_dir / "distributions.png")
    plt.close()
    
    # 9. Confidence Distribution, 10. Entropy Distribution
    if "department_confidence" in df.columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(df["department_confidence"], bins=50)
        plt.title("Confidence Distribution")
        plt.savefig(results_dir / "confidence_dist.png")
        plt.close()
        
    if "department_entropy" in df.columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(df["department_entropy"], bins=50)
        plt.title("Entropy Distribution")
        plt.savefig(results_dir / "entropy_dist.png")
        plt.close()
        
    # Errors and Corrects
    errors_df = df[df["ground_truth_department"] != df["predicted_department"]].copy()
    corrects_df = df[df["ground_truth_department"] == df["predicted_department"]].copy()
    
    # 17. Misclassified sample export, 18. Correct prediction export
    errors_df.to_csv(results_dir / "misclassified_samples.csv", index=False)
    corrects_df.to_csv(results_dir / "correct_samples.csv", index=False)
    
    # 11. Top-100 highest-confidence errors, 12. Top-100 lowest-confidence errors
    if "department_confidence" in errors_df.columns:
        errors_df = errors_df.sort_values(by="department_confidence", ascending=False)
        errors_df.head(100).to_csv(results_dir / "top_100_high_conf_errors.csv", index=False)
        
        errors_df = errors_df.sort_values(by="department_confidence", ascending=True)
        errors_df.head(100).to_csv(results_dir / "top_100_low_conf_errors.csv", index=False)
        
    # 13. Most confused class pairs
    pairs = []
    for t_idx, t_cls in enumerate(classes):
        for p_idx, p_cls in enumerate(classes):
            if t_cls != p_cls:
                count = cm[t_idx, p_idx]
                if count > 0:
                    pairs.append({"True": t_cls, "Predicted": p_cls, "Count": count})
                    
    pairs_df = pd.DataFrame(pairs).sort_values("Count", ascending=False)
    pairs_df.to_csv(results_dir / "most_confused_pairs.csv", index=False)
    
    # 14. Classes never predicted, 15. Classes never correctly predicted
    never_predicted = [c for c in classes if c not in pred_dist.index or pred_dist[c] == 0]
    
    correct_counts = corrects_df["ground_truth_department"].value_counts()
    never_correctly_predicted = [c for c in classes if c not in correct_counts.index or correct_counts[c] == 0]
    
    # 16. Per-class calibration
    calibration = []
    for cls in classes:
        cls_df = df[df["predicted_department"] == cls]
        if len(cls_df) > 0:
            acc = (cls_df["ground_truth_department"] == cls_df["predicted_department"]).mean()
            conf = cls_df["department_confidence"].mean() if "department_confidence" in cls_df.columns else 0
            calibration.append({"Class": cls, "Accuracy": acc, "Mean_Confidence": conf, "Support": len(cls_df)})
            
    calib_df = pd.DataFrame(calibration)
    if not calib_df.empty:
        calib_df.to_csv(results_dir / "per_class_calibration.csv", index=False)
        
    # Markdown Summary
    md = f"# Error Analysis Summary\n\n"
    md += f"## Classes never predicted\n"
    md += ", ".join(never_predicted) if never_predicted else "None"
    md += f"\n\n## Classes never correctly predicted\n"
    md += ", ".join(never_correctly_predicted) if never_correctly_predicted else "None"
    md += f"\n\n## Top Confused Pairs\n"
    if not pairs_df.empty:
        md += pairs_df.head(10).to_markdown(index=False)
    
    with open(results_dir / "summary.md", "w") as f:
        f.write(md)
        
    print(f"Error analysis completed. Results saved to {results_dir}")

if __name__ == '__main__':
    main()
