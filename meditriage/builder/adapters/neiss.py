import pandas as pd
from pathlib import Path
from .base import BaseAdapter

class NeissAdapter(BaseAdapter):
    @property
    def dataset_source(self): return "neiss"
    def ingest(self, raw_path: str) -> pd.DataFrame:
        p = Path(raw_path) / "neiss_all.parquet"
        if not p.exists(): return pd.DataFrame()
        df = pd.read_parquet(p)
        records = []
        for i, row in df.iterrows():
            text = str(row.get("Narrative", ""))
            if not text or text == "nan": text = str(row.get("narrative", ""))
            if not text or text == "nan": continue
            records.append({
                "tracking_id": f"neiss::{i}::0",
                "seed_id": f"neiss::{i}",
                "dataset_source": "neiss",
                "raw_text": text,
                "raw_medical_specialty": "Emergency",
                "raw_severity": str(row.get("Disposition", "")),
                "language": "en",
                "text": text,
                "department_code": "EMERGENCY",
                "routing_confidence": "high",
                "severity_label": "UNKNOWN",
                "severity_label_source": "native",
                "is_perturbed": False,
                "variant_index": 0,
                "split": None
            })
        return pd.DataFrame(records)
