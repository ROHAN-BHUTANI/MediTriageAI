import pandas as pd
from pathlib import Path
import zipfile
from .base import BaseAdapter

class L3cubeCodeMixedAdapter(BaseAdapter):
    @property
    def dataset_source(self): return "l3cube_code_mixed"
    def ingest(self, raw_path: str) -> pd.DataFrame:
        zp = Path(raw_path) / "code-mixed-nlp.zip"
        records = []
        if zp.exists():
            try:
                with zipfile.ZipFile(zp, 'r') as z:
                    for name in z.namelist():
                        if name.endswith('all.txt'):
                            with z.open(name) as f:
                                for i, line in enumerate(f):
                                    text = line.decode('utf-8').strip()
                                    if not text: continue
                                    records.append({
                                        "tracking_id": f"l3cube_code_mixed::{i}::0",
                                        "seed_id": f"l3cube_code_mixed::{i}",
                                        "dataset_source": "l3cube_code_mixed",
                                        "raw_text": text,
                                        "raw_medical_specialty": None,
                                        "raw_severity": None,
                                        "language": "hinglish",
                                        "text": text,
                                        "department_code": "UNKNOWN",
                                        "routing_confidence": "low",
                                        "severity_label": "UNKNOWN",
                                        "severity_label_source": "native",
                                        "is_perturbed": False,
                                        "variant_index": 0,
                                        "split": None
                                    })
            except: pass
        return pd.DataFrame(records)
