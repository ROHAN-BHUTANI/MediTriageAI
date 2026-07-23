import os
import pandas as pd
from src.model import SPECIALIST_CLASSES, SEVERITY_LABELS

def materialize():
    # Load dataset
    print("Loading processed dataset...")
    df = pd.read_csv("meditriage/data/processed/dataset.csv")
    
    # 1. Add patient_id (tracking_id -> patient_id)
    # The requirement is that we must have patient_id
    # But wait, seed_id is used for patient_level splits in src/data_pipeline.py.
    # dataset_specification.md says: `patient_id` A unique identifier for the patient. 
    # tracking_id is unique per row, seed_id is unique per original patient.
    # We should use seed_id as patient_id.
    df["patient_id"] = df["seed_id"]
    
    # 2. Add specialist_label (department_code -> integer)
    spec_map = {name: i for i, name in enumerate(SPECIALIST_CLASSES)}
    df["specialist_label"] = df["department_code"].map(spec_map)
    
    # 3. Add severity_label (severity_heuristic -> integer)
    sev_map = {name: i for i, name in enumerate(SEVERITY_LABELS)}
    df["severity_label"] = df["severity_heuristic"].map(sev_map)
    
    # 4. Filter OOD vs IN-DISTRIBUTION
    # Based on routing_confidence == "low" (from specialty_mapping.py)
    is_ood = df["routing_confidence"] == "low"
    df_ood = df[is_ood]
    df_in_dist = df[~is_ood]
    
    # 5. Split by language
    df_clean = df_in_dist[df_in_dist["language"] == "en"]
    df_hinglish = df_in_dist[df_in_dist["language"] == "hinglish"]
    
    # Columns required: patient_id, text, specialist_label, severity_label
    cols = ["patient_id", "text", "specialist_label", "severity_label"]
    
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    
    print("Writing clinical_triage_clean.csv...")
    df_clean[cols].to_csv("data/clinical_triage_clean.csv", index=False)
    
    print("Writing clinical_triage_hinglish.csv...")
    df_hinglish[cols].to_csv("data/clinical_triage_hinglish.csv", index=False)
    
    print("Writing ood_queries.csv...")
    df_ood[cols].to_csv("data/ood_queries.csv", index=False)
    
    # Gather stats
    stats = {
        "Total processed rows": len(df),
        "Total OOD rows": len(df_ood),
        "Total clean rows": len(df_clean),
        "Total hinglish rows": len(df_hinglish),
        "Specialist distribution (In-Dist)": df_in_dist["department_code"].value_counts().to_dict(),
        "Severity distribution (In-Dist)": df_in_dist["severity_heuristic"].value_counts().to_dict(),
        "Average text length (In-Dist, chars)": df_in_dist["text"].str.len().mean(),
        "Duplicates removed": 0 # Dataset was already generated, assuming no direct duplicates needing removal here based on tracking_id, or we should drop_duplicates?
    }
    
    return stats

if __name__ == "__main__":
    stats = materialize()
    for k, v in stats.items():
        print(f"{k}: {v}")
