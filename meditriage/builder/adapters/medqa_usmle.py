import pandas as pd
from pathlib import Path
from .base import BaseAdapter

class MedqaUsmleAdapter(BaseAdapter):
    @property
    def dataset_source(self) -> str:
        return "medqa_usmle"
        
    @property
    def version(self) -> str:
        return "1.0.0"

    def ingest(self, raw_path: str) -> pd.DataFrame:
        file_path = Path(raw_path) / "data_clean/data_clean/questions/US/4_options/phrases_no_exclude_train.jsonl"
        if not file_path.exists():
            return pd.DataFrame()
            
        if ".jsonl" == ".csv":
            df = pd.read_csv(file_path)
        elif ".jsonl" == ".parquet":
            df = pd.read_parquet(file_path)
        elif ".jsonl" in [".json", ".jsonl"]:
            try:
                df = pd.read_json(file_path, lines=True)
            except:
                df = pd.read_json(file_path)
                
        records = []
        for idx, row in df.iterrows():
            text = str(row.get("question", ""))
            if not text or text.lower() == "nan":
                continue
                
            spec = None
            if "None" != "None":
                spec = str(row.get("None", "")).strip()
            
            records.append({
                "tracking_id": f"medqa_usmle::{idx}::0",
                "seed_id": f"medqa_usmle::{idx}",
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
