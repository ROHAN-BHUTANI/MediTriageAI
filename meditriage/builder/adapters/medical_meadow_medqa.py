import pandas as pd
from pathlib import Path
from .base import BaseAdapter

class MedicalMeadowMedqaAdapter(BaseAdapter):
    @property
    def dataset_source(self): return "medical_meadow_medqa"
    def ingest(self, raw_path: str) -> pd.DataFrame:
        p = Path(raw_path) / "medical_meadow_medqa.json"
        if not p.exists(): return pd.DataFrame()
        df = pd.read_json(p)
        records = []
        for i, row in df.iterrows():
            text = str(row.get("input", ""))
            if not text or text == "nan": text = str(row.get("instruction", ""))
            if not text or text == "nan": continue
            records.append({
                "tracking_id": f"medical_meadow_medqa::{i}::0",
                "seed_id": f"medical_meadow_medqa::{i}",
                "dataset_source": "medical_meadow_medqa",
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
