import pandas as pd
from pathlib import Path
from typing import Iterator
from datetime import datetime, timezone
import pyarrow.parquet as pq
from .base import BaseAdapter

class NeissAdapter(BaseAdapter):
    """
    Adapter for the NEISS dataset.
    
    Mapping Strategy:
    - `Narrative_1` -> `raw_text`
    - Stream parquet using pyarrow.
    """
    @property
    def dataset_source(self) -> str:
        return "neiss"
        
    @property
    def version(self) -> str:
        return "1.0.0"

    def ingest(self, raw_path: str, chunk_size: int = 1000) -> Iterator[pd.DataFrame]:
        parquet_path = Path(raw_path) / "neiss_all.parquet"
        if not parquet_path.exists():
            return
            
        parquet_file = pq.ParquetFile(parquet_path)
        
        for batch in parquet_file.iter_batches(batch_size=chunk_size):
            chunk_df = batch.to_pandas()
            records = []
            
            for idx, row in chunk_df.iterrows():
                # Clean and extract
                text = str(row.get("Narrative_1", "")).strip()
                if not text or text.lower() == "nan":
                    continue
                    
                # Build record
                records.append({
                    "dataset_source": self.dataset_source,
                    "raw_text": text,
                    "department": "Unknown",
                    "triage_level": None,
                    "language": "en"
                })
                
            if records:
                yield pd.DataFrame(records)
