import pandas as pd
from pathlib import Path
from typing import Iterator
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

    def ingest(self, raw_path: str, chunk_size: int = 100000) -> Iterator[pd.DataFrame]:
        csv_path = Path(raw_path) / "PMC-Patients.csv"
        if not csv_path.exists():
            return
            
        for chunk_idx, chunk_df in enumerate(pd.read_csv(csv_path, chunksize=chunk_size)):
            # Vectorized operations
            chunk_df['patient'] = chunk_df['patient'].fillna('').astype(str).str.strip()
            valid_mask = (chunk_df['patient'] != '') & (chunk_df['patient'].str.lower() != 'nan')
            valid_df = chunk_df[valid_mask]
            
            if len(valid_df) == 0:
                continue
                
            out_df = pd.DataFrame({
                "dataset_source": self.dataset_source,
                "raw_text": valid_df["patient"],
                "department": None,
                "triage_level": None,
                "language": "en"
            })
            # Include legacy UUIDs so the new schema drops them later, or just don't include them
            # because Stage 3 validation doesn't care. The new schema just needs raw_text.
            out_df["id"] = [f"pmc_patients::{chunk_idx}::{i}" for i in range(len(valid_df))]
            
            yield out_df
