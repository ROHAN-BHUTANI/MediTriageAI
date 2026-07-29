import pandas as pd
from pathlib import Path
from .base import BaseAdapter

class KaggleMedicalTriageAdapter(BaseAdapter):
    @property
    def dataset_source(self): return "kaggle_medical_triage"
    def ingest(self, raw_path: str) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "tracking_id": "kaggle_medical_triage::0::0",
                "seed_id": "kaggle_medical_triage::0",
                "dataset_source": "kaggle_medical_triage",
                "raw_text": "Sample kaggle triage note",
                "raw_medical_specialty": "General",
                "raw_severity": "Urgent",
                "language": "en",
                "text": "Sample kaggle triage note",
                "department_code": "GENERAL",
                "routing_confidence": "high",
                "severity_label": "HIGH",
                "severity_label_source": "native",
                "is_perturbed": False,
                "variant_index": 0,
                "split": None
            }
        ])
