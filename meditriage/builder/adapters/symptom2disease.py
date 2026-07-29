import pandas as pd
from pathlib import Path
from .base import BaseAdapter

class Symptom2diseaseAdapter(BaseAdapter):
    @property
    def dataset_source(self): return "symptom2disease"
    def ingest(self, raw_path: str) -> pd.DataFrame:
        p = Path(raw_path) / "Symptom2Disease.csv"
        if not p.exists(): return pd.DataFrame()
        df = pd.read_csv(p)
        records = []
        for i, row in df.iterrows():
            text = str(row.get("text", ""))
            if not text or text == "nan": continue
            label = str(row.get("label", ""))
            records.append({
                "tracking_id": f"symptom2disease::{i}::0",
                "seed_id": f"symptom2disease::{i}",
                "dataset_source": "symptom2disease",
                "raw_text": text,
                "raw_medical_specialty": label,
                "raw_severity": None,
                "language": "en",
                "text": text,
                "department_code": label,
                "routing_confidence": "high",
                "severity_label": "UNKNOWN",
                "severity_label_source": "native",
                "is_perturbed": False,
                "variant_index": 0,
                "split": None
            })
        return pd.DataFrame(records)
