import pandas as pd
import json
from pathlib import Path
import zipfile
from .base import BaseAdapter

class MedqaUsmleAdapter(BaseAdapter):
    @property
    def dataset_source(self): return "medqa_usmle"
    def ingest(self, raw_path: str) -> pd.DataFrame:
        zp = Path(raw_path) / "data_clean.zip"
        records = []
        if zp.exists():
            with zipfile.ZipFile(zp, 'r') as z:
                for name in z.namelist():
                    if name.endswith('.jsonl'):
                        with z.open(name) as f:
                            for line in f:
                                try:
                                    data = json.loads(line.decode('utf-8', errors='ignore'))
                                except: continue
                                text = data.get("question", "")
                                if not text: continue
                                records.append({
                                    "tracking_id": f"medqa_usmle::{len(records)}::0",
                                    "seed_id": f"medqa_usmle::{len(records)}",
                                    "dataset_source": "medqa_usmle",
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
