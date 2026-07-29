import pandas as pd
from pathlib import Path
from .base import BaseAdapter

class ChatdoctorIcliniqAdapter(BaseAdapter):
    @property
    def dataset_source(self): return "chatdoctor_icliniq"
    def ingest(self, raw_path: str) -> pd.DataFrame:
        p = list(Path(raw_path).rglob("*.parquet"))
        if not p: return pd.DataFrame()
        df = pd.read_parquet(p[0])
        records = []
        for i, row in df.iterrows():
            text = str(row.get("input", ""))
            if not text or text == "nan": continue
            records.append({
                "tracking_id": f"chatdoctor_icliniq::{i}::0",
                "seed_id": f"chatdoctor_icliniq::{i}",
                "dataset_source": "chatdoctor_icliniq",
                "raw_text": text,
                "raw_medical_specialty": None,
                "raw_severity": None,
                "language": "en",
                "text": text,
                "department_code": "UNKNOWN",
                "routing_confidence": "low",
                "severity_label": "UNKNOWN",
                "severity_label_source": "native",
                "is_perturbed": False,
                "variant_index": 0,
                "split": None
            })
        return pd.DataFrame(records)
