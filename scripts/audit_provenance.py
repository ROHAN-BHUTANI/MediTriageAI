import sys
sys.path.insert(0, ".")
import pandas as pd
from src.dataset import SPECIALIST_CLASSES, SEVERITY_LABELS

def audit_provenance(path: str):
    df = pd.read_parquet(path)
    
    # Clean up department logic the same way schema.py does
    if "department" in df.columns:
        df["department"] = df["department"].replace({"Emergency": "ED"})
        
    results = []
    
    for source, group in df.groupby("dataset_source"):
        total = len(group)
        
        has_dept = group["department"].notna()
        valid_dept = group["department"].isin(SPECIALIST_CLASSES) & has_dept
        
        has_triage = group["triage_level"].notna()
        mapped_triage = group["triage_level"].astype(str).copy()
        
        is_num = mapped_triage.apply(lambda x: isinstance(x, (float, int)) or (isinstance(x, str) and x.replace('.', '').isdigit()))
        mapped_triage.loc[is_num & has_triage] = "S" + pd.to_numeric(mapped_triage.loc[is_num & has_triage], errors='coerce').fillna(0).astype(int).astype(str)
        
        valid_triage = mapped_triage.isin(SEVERITY_LABELS) & has_triage
        both_valid = valid_dept & valid_triage
        
        retention = (both_valid.sum() / total) * 100 if total > 0 else 0
        
        results.append({
            "Adapter": source,
            "Total": total,
            "Department": int(valid_dept.sum()),
            "Severity": int(valid_triage.sum()),
            "Both": int(both_valid.sum()),
            "Retention %": f"{retention:.2f}%"
        })
        
    print("| Adapter | Total | Department | Severity | Both | Retention % |")
    print("|---|---|---|---|---|---|")
    for r in results:
        print(f"| {r['Adapter']} | {r['Total']} | {r['Department']} | {r['Severity']} | {r['Both']} | {r['Retention %']} |")
        
if __name__ == "__main__":
    audit_provenance("meditriage/data/processed/dataset.parquet")
