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

    def ingest(self, raw_path: str, chunk_size: int = 100000) -> Iterator[pd.DataFrame]:
        parquet_path = Path(raw_path) / "neiss_all.parquet"
        if not parquet_path.exists():
            return
            
        parquet_file = pq.ParquetFile(parquet_path)
        
        for batch in parquet_file.iter_batches(batch_size=chunk_size):
            chunk_df = batch.to_pandas()
            
            # Vectorized operations
            # Filter valid text
            chunk_df['Narrative_1'] = chunk_df['Narrative_1'].astype(str).str.strip()
            valid_mask = (chunk_df['Narrative_1'] != '') & (chunk_df['Narrative_1'].str.lower() != 'nan')
            valid_df = chunk_df[valid_mask]
            
            if len(valid_df) == 0:
                continue
                
            # Create standard dataframe
            out_df = pd.DataFrame({
                "dataset_source": self.dataset_source,
                "raw_text": valid_df["Narrative_1"],
                "department": "Unknown",
                "triage_level": None,
                "language": "en"
            })
            
            yield out_df
