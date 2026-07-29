import pandas as pd
from pathlib import Path
from typing import Iterator
from datetime import datetime, timezone
import pyarrow.parquet as pq
from .base import BaseAdapter

class ChatDoctorIcliniqAdapter(BaseAdapter):
    """
    Adapter for the ChatDoctor iCliniq dataset.
    
    Mapping Strategy:
    - `input` -> `raw_text`
    - Stream parquet using pyarrow.
    """
    @property
    def dataset_source(self) -> str:
        return "chatdoctor_icliniq"
        
    @property
    def version(self) -> str:
        return "1.0.0"

    def ingest(self, raw_path: str, chunk_size: int = 1000) -> Iterator[pd.DataFrame]:
        data_dir = Path(raw_path) / "data"
        if not data_dir.exists():
            return
            
        parquet_files = list(data_dir.glob("*.parquet"))
        if not parquet_files:
            return
            
        for pfile in parquet_files:
            parquet_file = pq.ParquetFile(pfile)
            
            for batch in parquet_file.iter_batches(batch_size=chunk_size):
                chunk_df = batch.to_pandas()
                records = []
                
                for idx, row in chunk_df.iterrows():
                    # Clean and extract
                    text = str(row.get("input", "")).strip()
                    if not text or text.lower() == "nan":
                        continue
                        
                    # Build record
                    records.append({
                        "tracking_id": f"chatdoctor_icliniq::{pfile.name}::{idx}::0",
                        "seed_id": f"chatdoctor_icliniq::{pfile.name}::{idx}",
                        "dataset_source": self.dataset_source,
                        "raw_text": text,
                        "raw_medical_specialty": None,
                        "raw_severity": None,
                        "language": "en",
                        "text": text,
                        "department": None,
                        "routing_confidence": "low",
                        "triage_level": None,
                        "severity_label_source": "native",
                        "is_perturbed": False,
                        "variant_index": 0,
                        "split": None,
                        "extraction_timestamp": datetime.now(timezone.utc).isoformat(),
                        "original_schema_version": self.version
                    })
                    
                if records:
                    yield pd.DataFrame(records)
