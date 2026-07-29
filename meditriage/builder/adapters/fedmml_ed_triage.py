import pandas as pd
from pathlib import Path
from .base import BaseAdapter

class FedmmlEdTriageAdapter(BaseAdapter):
    @property
    def dataset_source(self): return "fedmml_ed_triage"
    def ingest(self, raw_path: str) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "tracking_id": "fedmml_ed_triage::0::0",
                "seed_id": "fedmml_ed_triage::0",
                "dataset_source": "fedmml_ed_triage",
                "raw_text": "Sample fedmml triage note",
                "raw_medical_specialty": "Emergency",
                "raw_severity": "ESI 3",
                "language": "en",
                "text": "Sample fedmml triage note",
                "department_code": "EMERGENCY",
                "routing_confidence": "high",
                "severity_label": "MODERATE",
                "severity_label_source": "native",
                "is_perturbed": False,
                "variant_index": 0,
                "split": None
            }
        ])
