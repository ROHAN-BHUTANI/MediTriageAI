import pandas as pd
from pathlib import Path
from .base import BaseAdapter

class MtsamplesAdapter(BaseAdapter):
    @property
    def dataset_source(self): return "mtsamples"
    def ingest(self, raw_path: str) -> pd.DataFrame:
        p = Path(raw_path) / "mtsamples (1).csv"
        if not p.exists(): return pd.DataFrame()
        df = pd.read_csv(p)
        records = []
        for i, row in df.iterrows():
            text = str(row.get("transcription", ""))
            if not text or text == "nan": continue
            spec = str(row.get("medical_specialty", ""))
            records.append({
                "tracking_id": f"mtsamples::{i}::0",
                "seed_id": f"mtsamples::{i}",
                "dataset_source": "mtsamples",
                "raw_text": text,
                "raw_medical_specialty": spec,
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
