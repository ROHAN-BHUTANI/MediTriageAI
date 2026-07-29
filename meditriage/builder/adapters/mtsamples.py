import pandas as pd
from pathlib import Path
from .base import BaseAdapter

class MtsamplesAdapter(BaseAdapter):
    @property
    def dataset_source(self) -> str:
        return "mtsamples"
        
    @property
    def version(self) -> str:
        return "1.0.0"

    def ingest(self, raw_path: str) -> pd.DataFrame:
        file_path = Path(raw_path) / "mtsamples (1).csv"
        if not file_path.exists():
            return pd.DataFrame()
            
        if ".csv" == ".csv":
            df = pd.read_csv(file_path)
        elif ".csv" == ".parquet":
            df = pd.read_parquet(file_path)
        elif ".csv" in [".json", ".jsonl"]:
            try:
                df = pd.read_json(file_path, lines=True)
            except:
                df = pd.read_json(file_path)
                
        records = []
        for idx, row in df.iterrows():
            text = str(row.get("transcription", ""))
            if not text or text.lower() == "nan":
                continue
                
            spec = None
            if "medical_specialty" != "None":
                spec = str(row.get("medical_specialty", "")).strip()
            
            records.append({
                "tracking_id": f"mtsamples::{idx}::0",
                "seed_id": f"mtsamples::{idx}",
                "dataset_source": self.dataset_source,
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
