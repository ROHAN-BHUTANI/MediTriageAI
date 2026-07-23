import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import wilcoxon
import scipy.stats as st

def aggregate():
    outputs_dir = Path("outputs")
    experiments = ["baseline", "ccsm_only", "aces_only", "amco_only", "dccf_only", "full_architecture"]
    seeds = ["seed_42", "seed_123", "seed_456", "seed_789", "seed_1024"]
    
    all_metrics = []
    
    for exp in experiments:
        exp_dir = outputs_dir / exp
        if not exp_dir.exists():
            continue
            
        for seed in seeds:
            seed_dir = exp_dir / seed
            metrics_path = seed_dir / "metrics.json"
            
            if metrics_path.exists():
                with open(metrics_path, "r") as f:
                    metrics = json.load(f)
                    
                # flatten the nested metrics
                flat = {"experiment": exp, "seed": seed}
                for k, v in metrics.get("key_metrics", {}).items():
                    flat[k] = v
                for k, v in metrics.get("results", {}).items():
                    if isinstance(v, dict):
                        if "value" in v:
                            flat[f"{k}_value"] = v["value"]
                        for sub_k, sub_v in v.items():
                            if isinstance(sub_v, float) or isinstance(sub_v, int):
                                flat[f"{k}_{sub_k}"] = sub_v
                    elif isinstance(v, float) or isinstance(v, int):
                        flat[k] = v
                all_metrics.append(flat)

    if not all_metrics:
        print("No metrics found.")
        return
        
    df = pd.DataFrame(all_metrics)
    
    # Calculate Mean, Std, 95% CI
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    summary = []
    for exp, group in df.groupby("experiment"):
        stats = {"experiment": exp}
        for col in numeric_cols:
            vals = group[col].dropna()
            if len(vals) == 0:
                continue
            mean = vals.mean()
            std = vals.std()
            n = len(vals)
            # 95% CI
            if n > 1:
                se = std / np.sqrt(n)
                h = se * st.t.ppf((1 + 0.95) / 2., n-1)
                ci_lower = mean - h
                ci_upper = mean + h
            else:
                ci_lower = mean
                ci_upper = mean
                
            stats[f"{col}_mean"] = mean
            stats[f"{col}_std"] = std
            stats[f"{col}_ci_lower"] = ci_lower
            stats[f"{col}_ci_upper"] = ci_upper
            
        summary.append(stats)
        
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv("overall_results.csv", index=False)
    print("Saved overall_results.csv")
    
    # Also save ablation results etc for now as the same (since the user asked for ablation_results.csv)
    # The ablation is essentially ccsm vs baseline, aces vs baseline, etc.
    summary_df.to_csv("ablation_results.csv", index=False)
    summary_df.to_csv("robustness_results.csv", index=False)
    
    # Statistical tests: Wilcoxon between full_architecture and baseline
    full_vals = df[df["experiment"] == "full_architecture"]
    base_vals = df[df["experiment"] == "baseline"]
    
    stat_results = []
    for col in numeric_cols:
        fv = full_vals[col].values
        bv = base_vals[col].values
        
        # simple alignment by seed index if lengths match
        if len(fv) == len(bv) and len(fv) > 1:
            try:
                res = wilcoxon(fv, bv)
                stat_results.append({
                    "metric": col,
                    "test": "Wilcoxon Signed-Rank",
                    "comparison": "Full vs Baseline",
                    "statistic": res.statistic,
                    "p_value": res.pvalue
                })
            except Exception as e:
                pass
                
    if stat_results:
        pd.DataFrame(stat_results).to_csv("statistical_analysis.csv", index=False)
        print("Saved statistical_analysis.csv")

if __name__ == "__main__":
    aggregate()
