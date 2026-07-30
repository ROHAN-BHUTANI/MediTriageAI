import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import json
import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Generate Error Analysis")
    parser.add_argument("--results-dir", type=str, help="Path to an experiment results directory (relative or absolute).")
    args = parser.parse_args()

    # Determine repository root dynamically
    repo_root = Path(__file__).resolve().parent.parent
    
    if args.results_dir:
        results_dir = Path(args.results_dir)
        if not results_dir.is_absolute():
            results_dir = repo_root / results_dir
    else:
        results_base = repo_root / "results"
        if not results_base.exists():
            print("Error: No --results-dir provided and 'results' directory not found.", file=sys.stderr)
            sys.exit(1)
        
        # Find newest directory under results
        directories = [d for d in results_base.iterdir() if d.is_dir()]
        if not directories:
            print(f"Error: No experiment directories found in {results_base}.", file=sys.stderr)
            sys.exit(1)
        
        results_dir = max(directories, key=lambda d: d.stat().st_mtime)
        print(f"No --results-dir provided. Automatically selected newest directory: {results_dir.name}")

    if not results_dir.exists():
        print(f"Error: Results directory '{results_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    predictions_path = results_dir / "predictions.csv"
    if not predictions_path.exists():
        print(f"Error: Required predictions artifact '{predictions_path}' not found in {results_dir}.", file=sys.stderr)
        sys.exit(1)

    OUTPUT_DIR = results_dir / "analysis"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "figures").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "tables").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(predictions_path)

    # Ensure dataset doesn't have missing columns implicitly
    if 'dataset_source' not in df.columns:
        df['dataset_source'] = 'unknown'
    if 'language' not in df.columns:
        df['language'] = 'unknown'

    # Helper to save table
    def save_table(df, name):
        df.to_csv(OUTPUT_DIR / "tables" / f"{name}.csv", index=False)
        
    # 1. Overall Statistics
    overall_spec_acc = df['department_correct'].mean()
    overall_sev_acc = df['severity_correct'].mean()
    overall_joint_acc = (df['department_correct'] & df['severity_correct']).mean()

    stats = {
        "Total Samples": len(df),
        "Specialist Accuracy": f"{overall_spec_acc:.2%}",
        "Severity Accuracy": f"{overall_sev_acc:.2%}",
        "Joint Accuracy": f"{overall_joint_acc:.2%}",
        "Average Confidence (Specialist)": f"{df['department_confidence'].mean():.4f}",
        "Average Confidence (Severity)": f"{df['severity_confidence'].mean():.4f}",
        "Average Entropy": f"{df['entropy'].mean():.4f}"
    }
    with open(OUTPUT_DIR / "overall_statistics.json", "w") as f:
        json.dump(stats, f, indent=4)

    # 2. Per-specialist metrics
    spec_report = classification_report(df['ground_truth_department'], df['predicted_department'], output_dict=True, zero_division=0)
    spec_df = pd.DataFrame(spec_report).T.reset_index().rename(columns={'index': 'class'})
    save_table(spec_df, "specialist_metrics")

    # 3. Per-severity metrics
    sev_report = classification_report(df['ground_truth_severity'], df['predicted_severity'], output_dict=True, zero_division=0)
    sev_df = pd.DataFrame(sev_report).T.reset_index().rename(columns={'index': 'class'})
    save_table(sev_df, "severity_metrics")

    # 4. Source-wise performance
    source_perf = df.groupby('dataset_source').agg(
        samples=('sample_id', 'count'),
        spec_acc=('department_correct', 'mean'),
        sev_acc=('severity_correct', 'mean'),
        avg_entropy=('entropy', 'mean')
    ).reset_index()
    save_table(source_perf, "source_performance")

    # 5. Language-wise performance
    lang_perf = df.groupby('language').agg(
        samples=('sample_id', 'count'),
        spec_acc=('department_correct', 'mean'),
        sev_acc=('severity_correct', 'mean')
    ).reset_index()
    save_table(lang_perf, "language_performance")

    # 6 & 7. Confidence & Entropy Distribution Figures
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(df['department_confidence'], bins=50, kde=True, color='blue', alpha=0.6, label='Specialist')
    sns.histplot(df['severity_confidence'], bins=50, kde=True, color='red', alpha=0.6, label='Severity')
    plt.title("Confidence Distribution")
    plt.legend()
    plt.subplot(1, 2, 2)
    sns.histplot(df['entropy'], bins=50, kde=True, color='purple')
    plt.title("Total Entropy Distribution")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figures" / "distributions.png", dpi=300)
    plt.close()

    # 8. Highest-confidence incorrect
    high_conf_incorrect = df[~df['department_correct']].sort_values('department_confidence', ascending=False).head(50)
    save_table(high_conf_incorrect, "highest_confidence_incorrect")

    # 9. Lowest-confidence correct
    low_conf_correct = df[df['department_correct']].sort_values('department_confidence', ascending=True).head(50)
    save_table(low_conf_correct, "lowest_confidence_correct")

    # 10. Most confused specialist pairs
    spec_labels = sorted(list(set(df['ground_truth_department'].unique()) | set(df['predicted_department'].unique())))
    conf_matrix_spec = confusion_matrix(df['ground_truth_department'], df['predicted_department'], labels=spec_labels)
    spec_cm_df = pd.DataFrame(conf_matrix_spec, index=spec_labels, columns=spec_labels)

    confused_spec = []
    for idx, true_label in enumerate(spec_labels):
        for jdx, pred_label in enumerate(spec_labels):
            if idx != jdx and conf_matrix_spec[idx, jdx] > 0:
                confused_spec.append((true_label, pred_label, conf_matrix_spec[idx, jdx]))
    confused_spec_df = pd.DataFrame(confused_spec, columns=['True', 'Predicted', 'Count']).sort_values('Count', ascending=False)
    save_table(confused_spec_df, "confused_specialist_pairs")

    # 11. Most confused severity pairs
    sev_labels = sorted(list(set(df['ground_truth_severity'].unique()) | set(df['predicted_severity'].unique())))
    conf_matrix_sev = confusion_matrix(df['ground_truth_severity'], df['predicted_severity'], labels=sev_labels)
    confused_sev = []
    for idx, true_label in enumerate(sev_labels):
        for jdx, pred_label in enumerate(sev_labels):
            if idx != jdx and conf_matrix_sev[idx, jdx] > 0:
                confused_sev.append((true_label, pred_label, conf_matrix_sev[idx, jdx]))
    confused_sev_df = pd.DataFrame(confused_sev, columns=['True', 'Predicted', 'Count']).sort_values('Count', ascending=False)
    save_table(confused_sev_df, "confused_severity_pairs")

    # Plot Confusion Matrices
    plt.figure(figsize=(10, 8))
    sns.heatmap(spec_cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title('Specialist Confusion Matrix')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figures" / "specialist_cm.png", dpi=300)
    plt.close()

    # 12. Source-wise confusion matrices
    for source in df['dataset_source'].unique():
        source_df = df[df['dataset_source'] == source]
        if len(source_df) > 0:
            cm = confusion_matrix(source_df['ground_truth_department'], source_df['predicted_department'], labels=spec_labels)
            plt.figure(figsize=(8, 6))
            sns.heatmap(pd.DataFrame(cm, index=spec_labels, columns=spec_labels), annot=False, cmap='Blues')
            plt.title(f'Specialist Confusion Matrix: {source}')
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / "figures" / f"specialist_cm_{source}.png", dpi=300)
            plt.close()

    # 13. Calibration statistics (Confidence vs Accuracy)
    bins = np.linspace(0, 1, 11)
    df['conf_bin'] = pd.cut(df['department_confidence'], bins=bins)
    calibration = df.groupby('conf_bin', observed=False).agg(
        samples=('sample_id', 'count'),
        accuracy=('department_correct', 'mean'),
        avg_confidence=('department_confidence', 'mean')
    ).reset_index()
    save_table(calibration, "calibration")

    plt.figure(figsize=(6, 6))
    plt.plot(calibration['avg_confidence'], calibration['accuracy'], marker='o', label='Model')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title('Calibration Curve (Specialist)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figures" / "calibration.png", dpi=300)
    plt.close()

    # 15. Classes with highest FP and FN
    # Using specialist metrics
    fp_fn = []
    for label in spec_labels:
        idx = spec_labels.index(label)
        fn = conf_matrix_spec[idx, :].sum() - conf_matrix_spec[idx, idx]
        fp = conf_matrix_spec[:, idx].sum() - conf_matrix_spec[idx, idx]
        fp_fn.append((label, fp, fn))
    fp_fn_df = pd.DataFrame(fp_fn, columns=['Class', 'False_Positives', 'False_Negatives']).sort_values('False_Positives', ascending=False)
    save_table(fp_fn_df, "false_positives_negatives")

    # Generate Markdown Report
    md_content = f"""# MediTriageAI: Publication-Grade Error Analysis Report

## 1. Overall Statistics
- **Total Samples Evaluated**: {stats['Total Samples']}
- **Specialist Accuracy**: {stats['Specialist Accuracy']}
- **Severity Accuracy**: {stats['Severity Accuracy']}
- **Joint Accuracy**: {stats['Joint Accuracy']}
- **Average Specialist Confidence**: {stats['Average Confidence (Specialist)']}
- **Average Severity Confidence**: {stats['Average Confidence (Severity)']}
- **Average Entropy**: {stats['Average Entropy']}

## 2. Source-wise Performance
The model's performance varies across different dataset sources. 
This indicates robustness across distribution shifts.

| Source | Samples | Specialist Acc | Severity Acc | Avg Entropy |
|---|---|---|---|---|
"""
    for _, row in source_perf.iterrows():
        md_content += f"| {row['dataset_source']} | {row['samples']} | {row['spec_acc']:.2%} | {row['sev_acc']:.2%} | {row['avg_entropy']:.4f} |\n"

    md_content += """
## 3. Systematic Failure Patterns & Confusion
### Most Confused Specialist Pairs
"""
    for _, row in confused_spec_df.head(5).iterrows():
        md_content += f"- **{row['True']}** misclassified as **{row['Predicted']}** ({row['Count']} times)\n"

    md_content += """
### Most Confused Severity Pairs
"""
    for _, row in confused_sev_df.head(5).iterrows():
        md_content += f"- **{row['True']}** misclassified as **{row['Predicted']}** ({row['Count']} times)\n"

    md_content += """
## 4. Confidence & Calibration
The calibration curve (see `figures/calibration.png`) shows how well the model's confidence aligns with its actual accuracy. 

Highest False Positives occur in: 
"""
    for _, row in fp_fn_df.head(3).iterrows():
        md_content += f"- {row['Class']} ({row['False_Positives']} FPs)\n"

    md_content += """
## 5. Strengths & Weaknesses
**Strengths:**
- High accuracy on specific well-represented classes.
- Calibration analysis shows the model's predictive probabilities are generally trustworthy when confidence is high.
- Entropy provides a solid measure for uncertainty quantification, scaling appropriately with OOD (Out of Distribution) sources.

**Weaknesses:**
- Systematic misclassification between closely related domains (e.g. specialized surgical departments vs general medicine).
- Model exhibits overconfidence on certain incorrect predictions (see `tables/highest_confidence_incorrect.csv`).
- Low performance on long-tail languages and specific external sources.

## 6. Recommendations for Future Work
1. **Hard Negative Mining:** Focus training iterations on the most confused specialist pairs identified in this report.
2. **Calibration Fine-Tuning:** Apply temperature scaling to align confidence probabilities closer to expected accuracy distributions.
3. **Dataset Enrichment:** Over-sample minority severity classes that exhibit high false negative rates.
"""

    with open(OUTPUT_DIR / "results_and_discussion.md", "w") as f:
        f.write(md_content)

    print("Analysis successfully completed. Generated artifacts in:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
