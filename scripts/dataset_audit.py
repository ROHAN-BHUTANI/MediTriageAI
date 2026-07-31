import os
import json
import pandas as pd
import numpy as np
from pathlib import Path

def main():
    results_dir = Path("results/dataset_audit")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    dataset_path = Path("meditriage/data/processed/dataset.parquet")
    if not dataset_path.exists():
        print("dataset.parquet not found!")
        return

    print("Loading dataset...")
    df = pd.read_parquet(dataset_path)
    
    audit = {}
    
    # Missing field statistics
    audit["missing_field_statistics"] = df.isnull().sum().to_dict()
    
    # Class counts
    if "department" in df.columns:
        dept_counts = df["department"].value_counts().to_dict()
        audit["class_counts_department"] = dept_counts
        
        # Imbalance ratio
        if len(dept_counts) > 1:
            max_class = max(dept_counts.values())
            min_class = min([v for v in dept_counts.values() if v > 0]) if any(v > 0 for v in dept_counts.values()) else 1
            audit["imbalance_ratio"] = max_class / min_class
    
    # Duplicate samples (exact row matches)
    audit["duplicate_samples"] = int(df.duplicated().sum())
    
    # Duplicated complaints & contradictory labels
    if "raw_text" in df.columns:
        audit["duplicated_complaints"] = int(df.duplicated(subset=["raw_text"]).sum())
        
        if "department" in df.columns:
            # identical complaint different department (contradictory labels)
            grouped = df.groupby("raw_text")["department"].nunique()
            contradictory = (grouped > 1).sum()
            audit["contradictory_labels_department"] = int(contradictory)
            audit["identical_complaint_different_department"] = int(contradictory)
            
        # Repeated templates (approximate by looking at high-frequency texts)
        text_counts = df["raw_text"].value_counts()
        repeated_templates = int((text_counts > 10).sum())
        audit["repeated_templates"] = repeated_templates
        
        # Text length statistics
        lengths = df["raw_text"].dropna().astype(str).str.len()
        audit["text_length_statistics"] = {
            "mean": float(lengths.mean()),
            "std": float(lengths.std()),
            "min": float(lengths.min()),
            "max": float(lengths.max()),
            "median": float(lengths.median())
        }
        
    # Repeated patients
    if "patient_id" in df.columns:
        patient_counts = df["patient_id"].value_counts()
        audit["repeated_patients"] = int((patient_counts > 1).sum())
    else:
        audit["repeated_patients"] = "No patient_id column"
        
    # Language statistics
    if "language" in df.columns:
        audit["language_statistics"] = df["language"].value_counts().to_dict()
        
    # Supervision coverage
    if "department" in df.columns and "triage_level" in df.columns:
        dept_supervised = df["department"].notnull().sum()
        triage_supervised = df["triage_level"].notnull().sum()
        audit["supervision_coverage"] = {
            "department_supervised": int(dept_supervised),
            "department_supervised_pct": float(dept_supervised / len(df)),
            "triage_supervised": int(triage_supervised),
            "triage_supervised_pct": float(triage_supervised / len(df)),
            "both_supervised": int((df["department"].notnull() & df["triage_level"].notnull()).sum())
        }
        
    # Save as JSON
    with open(results_dir / "audit_results.json", "w") as f:
        json.dump(audit, f, indent=4)
        
    # Save as Markdown
    md = "# Dataset Audit Results\n\n"
    for k, v in audit.items():
        md += f"## {k}\n"
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                md += f"- **{sub_k}**: {sub_v}\n"
        else:
            md += f"{v}\n"
        md += "\n"
        
    with open(results_dir / "audit_summary.md", "w") as f:
        f.write(md)
        
    print(f"Dataset audit completed. Results saved to {results_dir}")

if __name__ == '__main__':
    main()
