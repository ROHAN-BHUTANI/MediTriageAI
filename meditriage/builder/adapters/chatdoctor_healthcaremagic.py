import pandas as pd
from pathlib import Path
from typing import Iterator
from datetime import datetime, timezone
import pyarrow.parquet as pq
from .base import BaseAdapter

class ChatDoctorHealthcareMagicAdapter(BaseAdapter):
    """
    Adapter for the ChatDoctor HealthcareMagic dataset.
    
    Mapping Strategy:
    - `input` -> `raw_text`
    - Stream parquet using pyarrow.
    """
    @property
    def dataset_source(self) -> str:
        return "chatdoctor_healthcaremagic"
        
    @property
    def version(self) -> str:
        return "1.0.0"

    def ingest(self, raw_path: str, chunk_size: int = 100000) -> Iterator[pd.DataFrame]:
        data_dir = Path(raw_path) / "data"
        if not data_dir.exists():
            return
            
        parquet_files = list(data_dir.glob("*.parquet"))
        if not parquet_files:
            return
            
        for pfile in parquet_files:
            parquet_file = pq.ParquetFile(pfile)
            
            # Use iter_batches with batch_size=chunk_size
            for batch in parquet_file.iter_batches(batch_size=chunk_size):
                chunk_df = batch.to_pandas()
                
                # Vectorized operations
                chunk_df['input'] = chunk_df['input'].astype(str).str.strip()
                valid_mask = (chunk_df['input'] != '') & (chunk_df['input'].str.lower() != 'nan')
                valid_df = chunk_df[valid_mask]
                
                if len(valid_df) == 0:
                    continue
                    
                out_df = pd.DataFrame({
                    "dataset_source": self.dataset_source,
                    "raw_text": valid_df["input"],
                    "department": None,
                    "triage_level": None,
                    "language": "en"
                })
                out_df["id"] = [f"chatdoctor_healthcaremagic::{pfile.name}::{i}" for i in range(len(valid_df))]
                
                yield out_df
