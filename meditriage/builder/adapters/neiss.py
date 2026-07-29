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
            valid_df = chunk_df[valid_mask].copy()
            
            if len(valid_df) == 0:
                continue
                
            narrative_lower = valid_df['Narrative_1'].str.lower()
            
            # Rule-based mapping
            department = pd.Series(None, index=valid_df.index, dtype=object)
            
            # PEDIATRICS
            is_pediatric = (pd.to_numeric(valid_df['Age'], errors='coerce') < 18)
            department.loc[is_pediatric] = "PEDIATRICS"
            
            # DERM (Lacerations, burns, rashes)
            derm_mask = narrative_lower.str.contains('laceration|cut|burn|rash|skin|abrasion', regex=True)
            department.loc[derm_mask] = "DERM"
            
            # ORTHO (Fractures, sprains, bone injuries)
            ortho_mask = narrative_lower.str.contains('fracture|sprain|strain|bone|joint|knee|shoulder|ankle|wrist|hip|dislocation', regex=True)
            department.loc[ortho_mask] = "ORTHO"
            
            # NEURO (Head injuries, concussions)
            neuro_mask = narrative_lower.str.contains('head injury|concussion|headache|dizziness|seizure|loss of consciousness', regex=True)
            department.loc[neuro_mask] = "NEURO"
            
            # CARDIO_PULM (Chest pain, breathing)
            cardio_mask = narrative_lower.str.contains('chest pain|shortness of breath|asthma|heart|lung|breathing', regex=True)
            department.loc[cardio_mask] = "CARDIO_PULM"
            
            # OPHTHAL (Eye injuries)
            eye_mask = narrative_lower.str.contains('eye|cornea|vision', regex=True)
            department.loc[eye_mask] = "OPHTHAL"
            
            # ENT (Ear, nose, throat, foreign body in orifice)
            ent_mask = narrative_lower.str.contains('ear|nose|throat|swallowed', regex=True)
            department.loc[ent_mask] = "ENT"
            
            # Create standard dataframe
            out_df = pd.DataFrame({
                "dataset_source": self.dataset_source,
                "raw_text": valid_df["Narrative_1"],
                "department": department,
                "triage_level": None,
                "language": "en"
            })
            
            yield out_df
