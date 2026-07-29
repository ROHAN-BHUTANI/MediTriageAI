import pandas as pd
from pathlib import Path
from .base import BaseAdapter

class PMCPatientsAdapter(BaseAdapter):
    @property
    def dataset_source(self) -> str:
        return "pmc_patients"
        
    @property
    def version(self) -> str:
        return "1.0.0"

    def ingest(self, raw_path: str) -> pd.DataFrame:
        csv_path = Path(raw_path) / "PMC-Patients.csv"
        if not csv_path.exists():
            return pd.DataFrame()
            
        # PMC Patients is large, maybe limit for now if testing, but assume full
        df = pd.read_csv(csv_path) 
        
        records = []
        for idx, row in df.iterrows():
            text = str(row.get("text", ""))
            if not text:
                continue
                
            records.append({
                "tracking_id": f"pmc_patients::{idx}::0",
                "seed_id": f"pmc_patients::{idx}",
                "dataset_source": self.dataset_source,
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
