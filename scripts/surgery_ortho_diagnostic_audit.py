from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from sklearn.feature_extraction.text import TfidfVectorizer

REPO_ROOT = Path(__file__).resolve().parent.parent

def main():
    print("=" * 60)
    print("SURGERY-ORTHO FAST DIAGNOSTIC AUDIT")
    print("=" * 60)

    # 1. Output Directory
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = REPO_ROOT / "results" / "diagnostic_audit" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    latest_dir = REPO_ROOT / "results" / "diagnostic_audit" / "latest"
    if latest_dir.exists() or latest_dir.is_symlink():
        latest_dir.unlink(missing_ok=True)
    try:
        latest_dir.symlink_to(out_dir.name)
    except OSError:
        pass

    pred_dir = REPO_ROOT / "results" / "prediction_error_analysis" / "latest"
    
    mis_df = pd.read_csv(pred_dir / "misclassified_samples.csv")
    cor_df = pd.read_csv(pred_dir / "correct_predictions.csv")
    all_df = pd.concat([mis_df, cor_df], ignore_index=True)
    
    # 2. Extract Cohorts
    surgery_ortho = all_df[(all_df['true_specialist_name'] == 'SURGERY') & (all_df['predicted_specialist_name'] == 'ORTHO')]
    surgery_correct = all_df[(all_df['true_specialist_name'] == 'SURGERY') & (all_df['correctness_specialist'] == True)]
    ortho_correct = all_df[(all_df['true_specialist_name'] == 'ORTHO') & (all_df['correctness_specialist'] == True)]
    
    print(f"SURGERY->ORTHO Misclassified: {len(surgery_ortho)}")
    print(f"SURGERY Correctly Classified: {len(surgery_correct)}")
    print(f"ORTHO Correctly Classified: {len(ortho_correct)}")
    
    # 3. Assertions against official reports
    cm_df = pd.read_csv(pred_dir / "confusion_matrix.csv")
    cm_surgery = cm_df[cm_df['True_Class'] == 'SURGERY']
    assert not cm_surgery.empty, "SURGERY class not found in confusion matrix"
    cm_surgery_to_ortho = cm_surgery['ORTHO'].values[0]
    cm_surgery_to_surgery = cm_surgery['SURGERY'].values[0]
    
    per_class_df = pd.read_csv(pred_dir / "per_class_metrics.csv")
    pc_surgery = per_class_df[per_class_df['Class'] == 'SURGERY']
    assert not pc_surgery.empty, "SURGERY class not found in per_class_metrics"
    expected_recall = pc_surgery['Recall'].values[0]
    support = pc_surgery['Support'].values[0]
    computed_recall = len(surgery_correct) / support if support > 0 else 0
    
    print(f"Validation: Expected Recall={expected_recall:.4f}, Computed Recall={computed_recall:.4f}")
    assert len(surgery_ortho) == cm_surgery_to_ortho, f"Mismatch: SURGERY->ORTHO extracted {len(surgery_ortho)} vs CM {cm_surgery_to_ortho}"
    assert len(surgery_correct) == cm_surgery_to_surgery, f"Mismatch: SURGERY correct extracted {len(surgery_correct)} vs CM {cm_surgery_to_surgery}"
    assert abs(computed_recall - expected_recall) < 1e-4, f"Mismatch: Computed Recall {computed_recall} != Expected {expected_recall}"
    print("All validation assertions passed.")
    
    surgery_ortho.to_csv(out_dir / "surgery_to_ortho.csv", index=False)
    surgery_correct.to_csv(out_dir / "surgery_correct.csv", index=False)
    ortho_correct.to_csv(out_dir / "ortho_correct.csv", index=False)

    # 4. TF-IDF Analysis
    vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
    texts = pd.concat([surgery_ortho['original_complaint'], surgery_correct['original_complaint']])
    stats_df = pd.DataFrame()
    if len(texts) > 0:
        X = vectorizer.fit_transform(texts.fillna('')).toarray()
        feature_names = np.array(vectorizer.get_feature_names_out())
        
        X_mis = X[:len(surgery_ortho)]
        X_cor = X[len(surgery_ortho):]
        
        stats = []
        for i in range(X.shape[1]):
            mis_scores = X_mis[:, i]
            cor_scores = X_cor[:, i]
            
            if len(mis_scores) > 1 and len(cor_scores) > 1:
                t_stat, p_val = ttest_ind(mis_scores, cor_scores, equal_var=False)
            else:
                t_stat, p_val = 0.0, 1.0
                
            mean_mis = np.mean(mis_scores) if len(mis_scores) > 0 else 0.0
            mean_cor = np.mean(cor_scores) if len(cor_scores) > 0 else 0.0
            diff = mean_mis - mean_cor
            
            if p_val < 0.05 and not np.isnan(p_val):
                stats.append({
                    "token": feature_names[i],
                    "mean_tfidf_surgery_ortho": float(mean_mis),
                    "mean_tfidf_surgery_correct": float(mean_cor),
                    "diff": float(diff),
                    "t_stat": float(t_stat),
                    "p_value": float(p_val)
                })
                
        stats_df = pd.DataFrame(stats)
        if not stats_df.empty:
            stats_df = stats_df.sort_values(by="diff", ascending=False)
            stats_df.to_csv(out_dir / "tfidf_differences.csv", index=False)
    
    # 5. Length Analysis
    len_mis = surgery_ortho['original_complaint'].fillna('').str.len()
    len_cor = surgery_correct['original_complaint'].fillna('').str.len()
    if len(len_mis) > 1 and len(len_cor) > 1:
        t_stat_len, p_val_len = ttest_ind(len_mis, len_cor, equal_var=False)
    else:
        t_stat_len, p_val_len = 0.0, 1.0
    
    # 6. Routing Confidence Analysis
    route_mis = pd.to_numeric(surgery_ortho['routing_confidence'], errors='coerce').dropna()
    route_cor = pd.to_numeric(surgery_correct['routing_confidence'], errors='coerce').dropna()
    if len(route_mis) > 1 and len(route_cor) > 1:
        t_stat_route, p_val_route = ttest_ind(route_mis, route_cor, equal_var=False)
    else:
        t_stat_route, p_val_route = 0.0, 1.0
    
    # 7. Report Generation
    report = []
    report.append("# SURGERY-ORTHO Diagnostic Audit Report")
    report.append(f"**Generated:** {timestamp} UTC")
    report.append(f"\n## Cohort Summary")
    report.append(f"- **SURGERY -> ORTHO Misclassified (Shortcut):** {len(surgery_ortho)}")
    report.append(f"- **SURGERY Correct:** {len(surgery_correct)}")
    report.append(f"- **ORTHO Correct:** {len(ortho_correct)}")
    
    report.append(f"\n## Length Analysis")
    report.append(f"- **Mean Length (ORTHO Misclassified):** {np.mean(len_mis) if len(len_mis)>0 else 0:.1f} chars")
    report.append(f"- **Mean Length (SURGERY Correct):** {np.mean(len_cor) if len(len_cor)>0 else 0:.1f} chars")
    report.append(f"- **Significance (p-value):** {p_val_len:.4e}")
    if p_val_len < 0.05:
        report.append("  - *Conclusion: Text length is statistically different.*")
    
    report.append(f"\n## Routing Confidence Analysis")
    if len(route_mis) > 0 and len(route_cor) > 0:
        report.append(f"- **Mean Confidence (ORTHO Misclassified):** {np.mean(route_mis):.3f}")
        report.append(f"- **Mean Confidence (SURGERY Correct):** {np.mean(route_cor):.3f}")
        report.append(f"- **Significance (p-value):** {p_val_route:.4e}")
    
    report.append("\n## Top 15 Discriminative Tokens (TF-IDF)")
    if not stats_df.empty:
        report.append("| Token | Mean (ORTHO Shortcut) | Mean (SURGERY Correct) | Diff | p-value |")
        report.append("|---|---|---|---|---|")
        for _, row in stats_df.head(15).iterrows():
            report.append(f"| {row['token']} | {row['mean_tfidf_surgery_ortho']:.4f} | {row['mean_tfidf_surgery_correct']:.4f} | {row['diff']:.4f} | {row['p_value']:.4e} |")
    else:
        report.append("No statistically significant tokens found.")
        
    report.append("\n## Conclusion: Shortcut Learning Hypothesis")
    if not stats_df.empty and stats_df.iloc[0]['diff'] > 0.05:
        report.append("The hypothesis is **PROVEN**. Specific tokens (e.g. above) are spuriously driving SURGERY samples into the ORTHO prediction space.")
    else:
        report.append("The hypothesis is **REJECTED**. There are no dominant shortcut tokens with statistical significance separating the cohorts.")
        
    with open(out_dir / "surgery_ortho_report.md", "w") as f:
        f.write('\n'.join(report))
        
    print(f"Audit completed. Results saved to {out_dir}")

if __name__ == '__main__':
    main()
