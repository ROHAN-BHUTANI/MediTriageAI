import pandas as pd
from pathlib import Path
from .base import BaseAdapter

class MeddialogEnAdapter(BaseAdapter):
    @property
    def dataset_source(self): return "meddialog_en"
    def ingest(self, raw_path: str) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "tracking_id": "meddialog_en::0::0",
                "seed_id": "meddialog_en::0",
                "dataset_source": "meddialog_en",
                "raw_text": "Sample meddialog text",
                "raw_medical_specialty": "General",
                "raw_severity": None,
                "language": "en",
                "text": "Sample meddialog text",
                "department_code": "GENERAL",
                "routing_confidence": "high",
                "severity_label": "UNKNOWN",
                "severity_label_source": "native",
                "is_perturbed": False,
                "variant_index": 0,
                "split": None
            }
        ])
