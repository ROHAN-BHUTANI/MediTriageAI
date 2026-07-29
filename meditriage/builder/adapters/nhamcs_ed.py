import pandas as pd
from pathlib import Path
import zipfile
from .base import BaseAdapter

class NhamcsEdAdapter(BaseAdapter):
    @property
    def dataset_source(self): return "nhamcs_ed"
    def ingest(self, raw_path: str) -> pd.DataFrame:
        zips = list(Path(raw_path).glob("*.zip"))
        records = []
        for z in zips:
            try:
                with zipfile.ZipFile(z, 'r') as zf:
                    for name in zf.namelist():
                        if name.endswith('.csv'):
                            with zf.open(name) as f:
                                df = pd.read_csv(f)
                                for i, row in df.iterrows():
                                    text = str(row.get("RFV1", ""))
                                    if not text or text == "nan": continue
                                    records.append({
                                        "tracking_id": f"nhamcs_ed::{len(records)}::0",
                                        "seed_id": f"nhamcs_ed::{len(records)}",
                                        "dataset_source": "nhamcs_ed",
                                        "raw_text": text,
                                        "raw_medical_specialty": "Emergency",
                                        "raw_severity": None,
                                        "language": "en",
                                        "text": text,
                                        "department_code": "EMERGENCY",
                                        "routing_confidence": "high",
                                        "severity_label": "UNKNOWN",
                                        "severity_label_source": "native",
                                        "is_perturbed": False,
                                        "variant_index": 0,
                                        "split": None
                                    })
            except: pass
        return pd.DataFrame(records)
