import pandas as pd
from pathlib import Path
from typing import Iterator
from datetime import datetime, timezone
from .base import BaseAdapter

class MedqaUsmleAdapter(BaseAdapter):
    """
    Adapter for the MedQA USMLE dataset.
    
    Mapping Strategy:
    - `question` -> `raw_text`
    - Read from `data_clean/data_clean/questions/US/US_qbank.jsonl`.
    """
    @property
    def dataset_source(self) -> str:
        return "medqa_usmle"
        
    @property
    def version(self) -> str:
        return "1.0.0"

    def ingest(self, raw_path: str, chunk_size: int = 1000) -> Iterator[pd.DataFrame]:
        jsonl_path = Path(raw_path) / "data_clean" / "data_clean" / "questions" / "US" / "US_qbank.jsonl"
        if not jsonl_path.exists():
            return
            
        for chunk_idx, chunk_df in enumerate(pd.read_json(jsonl_path, lines=True, chunksize=chunk_size)):
            records = []
            
            for idx, row in chunk_df.iterrows():
                # Clean and extract
                text = str(row.get("question", "")).strip()
                if not text or text.lower() == "nan":
                    continue
                    
                # Build record
                records.append({
                    "tracking_id": f"medqa_usmle::{idx}::0",
                    "seed_id": f"medqa_usmle::{idx}",
                    "dataset_source": self.dataset_source,
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
                    "split": None,
                    "extraction_timestamp": datetime.now(timezone.utc).isoformat(),
                    "original_schema_version": self.version
                })
                
            if records:
                yield pd.DataFrame(records)
