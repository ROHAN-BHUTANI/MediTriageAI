from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
DATASET_PATH = REPO_ROOT / "meditriage" / "data" / "processed" / "dataset.csv"

# Load a quick snippet of the processed CSV to verify columns and data fields.
df = pd.read_csv(DATASET_PATH)

print("=" * 70)
print("MEDITRIAGEAI: HIGH-INTEGRITY DATASET SAMPLE")
print("=" * 70)

# Pull a sample that highlights a non-English perturbation variant.
sample_records = df[df["is_perturbed"]].head(2)

for idx, row in sample_records.iterrows():
    print(f"[Tracking ID: {row['tracking_id']}] | Split: {row['split'].upper()}")
    print(f"  Raw Specialty : {row['raw_medical_specialty']}")
    print(f"  Language      : {row['language']}")
    print(f"  Severity      : {row['severity_heuristic']} ({row['severity_label_source']})")
    print(f"  Text Snippet  : {str(row['text'])[:120]}...")
    print("-" * 70)