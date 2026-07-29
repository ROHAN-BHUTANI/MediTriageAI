import pandas as pd
from pathlib import Path
from typing import Iterator
from datetime import datetime, timezone
import sys
import os

# Add repo root to sys.path to allow importing from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from src.specialty_mapping import RAW_TO_DEPARTMENT
from .base import BaseAdapter

class MTSamplesAdapter(BaseAdapter):
    """
    Adapter for the MTSamples dataset.
    
    Mapping Strategy:
    - `transcription` -> `raw_text`. Fallback to `description` if `transcription` is empty or NaN.
    - `medical_specialty` -> `raw_medical_specialty`.
    - Drop records where the resulting text is empty.
    """
    @property
    def dataset_source(self) -> str:
        return "mtsamples"
        
    @property
    def version(self) -> str:
        return "1.1.0"

    def ingest(self, raw_path: str, chunk_size: int = 1000) -> Iterator[pd.DataFrame]:
        csv_path = Path(raw_path) / "mtsamples (1).csv"
        if not csv_path.exists():
            return
            
        for chunk_idx, chunk_df in enumerate(pd.read_csv(csv_path, index_col=0, chunksize=chunk_size)):
            records = []
            
            for idx, row in chunk_df.iterrows():
                # Clean and extract
                text = str(row.get("transcription", "")).strip()
                if not text or text.lower() == "nan":
                    text = str(row.get("description", "")).strip()
                    
                if not text or text.lower() == "nan":
                    continue # Drop completely empty records
                    
                specialty = str(row.get("medical_specialty", "")).strip().lower()
                if specialty == "nan" or not specialty:
                    continue
                
                # Canonical mapping using src.specialty_mapping
                # Create a case-insensitive map
                mapping = {k.lower(): v for k, v in RAW_TO_DEPARTMENT.items()}
                department = mapping.get(specialty, "GEN_MED") # Fallback to GEN_MED for unknown
                
                # Build record
                records.append({
                    "tracking_id": f"mtsamples::{idx}::0",
                    "seed_id": f"mtsamples::{idx}",
                    "dataset_source": self.dataset_source,
                    "raw_text": text,
                    "department": department,
                    "triage_level": None,
                    "language": "en",
                    "text": text,
                    "routing_confidence": "low",
                    "severity_label_source": "native",
                    "is_perturbed": False,
                    "variant_index": 0,
                    "split": None,
                    "extraction_timestamp": datetime.now(timezone.utc).isoformat(),
                    "original_schema_version": self.version
                })
                
            if records:
                yield pd.DataFrame(records)
