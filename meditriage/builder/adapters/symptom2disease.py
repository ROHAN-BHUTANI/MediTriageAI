import pandas as pd
from pathlib import Path
from typing import Iterator
from datetime import datetime, timezone
from .base import BaseAdapter

class Symptom2DiseaseAdapter(BaseAdapter):
    """
    Adapter for the Symptom2Disease dataset.
    
    Mapping Strategy:
    - `text` -> `raw_text`
    - `label` -> `raw_medical_specialty`
    """
    @property
    def dataset_source(self) -> str:
        return "symptom2disease"
        
    @property
    def version(self) -> str:
        return "1.0.0"

    def ingest(self, raw_path: str, chunk_size: int = 1000) -> Iterator[pd.DataFrame]:
        csv_path = Path(raw_path) / "Symptom2Disease.csv"
        if not csv_path.exists():
            return
            
        for chunk_idx, chunk_df in enumerate(pd.read_csv(csv_path, chunksize=chunk_size)):
            records = []
            
            for idx, row in chunk_df.iterrows():
                # Clean and extract
                text = str(row.get("text", "")).strip()
                if not text or text.lower() == "nan":
                    continue
                    
                label = str(row.get("label", "")).strip()
                if label.lower() == "nan":
                    label = None
                    
                # Build record
                records.append({
                    "tracking_id": f"symptom2disease::{idx}::0",
                    "seed_id": f"symptom2disease::{idx}",
                    "dataset_source": self.dataset_source,
                    "raw_text": text,
                    "raw_medical_specialty": label,
                    "raw_severity": None,
                    "language": "en",
                    "text": text,
                    "department_code": "UNKNOWN",
                    "routing_confidence": "low",
                    "severity_label": "UNKNOWN",
                    "severity_label_source": "native",
                    "is_perturbed": False,
                    "variant_index": 0,
                    "split": None,
                    "extraction_timestamp": datetime.now(timezone.utc).isoformat(),
                    "original_schema_version": self.version
                })
                
            if records:
                yield pd.DataFrame(records)
