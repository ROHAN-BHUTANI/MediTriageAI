import pandas as pd
from pathlib import Path
from .base import BaseAdapter

class MTSamplesAdapter(BaseAdapter):
    @property
    def dataset_source(self) -> str:
        return "mtsamples"
        
    @property
    def version(self) -> str:
        return "1.0.0"

    def ingest(self, raw_path: str) -> pd.DataFrame:
        csv_path = Path(raw_path) / "mtsamples (1).csv"
        if not csv_path.exists():
            return pd.DataFrame() # Fallback for tests if needed, or raise
            
        df = pd.read_csv(csv_path, index_col=0)
        
        records = []
        for idx, row in df.iterrows():
            text = str(row.get("transcription", ""))
            if not text or text.lower() == "nan":
                text = str(row.get("description", ""))
            
            records.append({
                "tracking_id": f"mtsamples::{idx}::0",
                "seed_id": f"mtsamples::{idx}",
                "dataset_source": self.dataset_source,
                "raw_text": text,
                "raw_medical_specialty": str(row.get("medical_specialty", "")).strip(),
                "raw_severity": None,
                "language": "en",
                "text": text,
                "department_code": "UNKNOWN", # populated later
                "routing_confidence": "low",  # populated later
                "severity_label": "UNKNOWN",  # populated later
                "severity_label_source": "native",
                "is_perturbed": False,
                "variant_index": 0,
                "split": None
            })
            
        return pd.DataFrame(records)
