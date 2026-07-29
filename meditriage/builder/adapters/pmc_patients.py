import pandas as pd
from pathlib import Path
from .base import BaseAdapter

class PmcPatientsAdapter(BaseAdapter):
    @property
    def dataset_source(self): return "pmc_patients"
    def ingest(self, raw_path: str) -> pd.DataFrame:
        p = Path(raw_path) / "PMC-Patients.csv"
        if not p.exists(): return pd.DataFrame()
        df = pd.read_csv(p)
        records = []
        for i, row in df.iterrows():
            text = str(row.get("patient", ""))
            if not text or text == "nan": continue
            records.append({
                "tracking_id": f"pmc_patients::{i}::0",
                "seed_id": f"pmc_patients::{i}",
                "dataset_source": "pmc_patients",
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
