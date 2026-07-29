import pandas as pd
from pathlib import Path
from typing import Iterator
from datetime import datetime, timezone
from .base import BaseAdapter

class PMCPatientsAdapter(BaseAdapter):
    """
    Adapter for the PMC Patients dataset.
    
    Mapping Strategy:
    - `patient` -> `raw_text`
    - Drop records with empty or NaN patient strings.
    - Yield chunks using chunksize to handle the 544MB file.
    """
    @property
    def dataset_source(self) -> str:
        return "pmc_patients"
        
    @property
    def version(self) -> str:
        return "1.1.0"

    def ingest(self, raw_path: str, chunk_size: int = 1000) -> Iterator[pd.DataFrame]:
        csv_path = Path(raw_path) / "PMC-Patients.csv"
        if not csv_path.exists():
            return
            
        for chunk_idx, chunk_df in enumerate(pd.read_csv(csv_path, chunksize=chunk_size)):
            records = []
            
            for idx, row in chunk_df.iterrows():
                # Clean and extract
                text = str(row.get("patient", "")).strip()
                if not text or text.lower() == "nan":
                    continue
                    
                # Build record
                records.append({
                    "tracking_id": f"pmc_patients::{idx}::0",
                    "seed_id": f"pmc_patients::{idx}",
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
