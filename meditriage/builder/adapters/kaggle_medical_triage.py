import pandas as pd
from typing import Iterator
from meditriage.builder.adapters.base import BaseAdapter

class KaggleMedicalTriageAdapter(BaseAdapter):
    """
    Adapter for Kaggle Medical Triage dataset.
    """
    
    @property
    def dataset_source(self) -> str:
        return "kaggle_medical_triage"
        
    @property
    def version(self) -> str:
        return "1.0"
        
    def ingest(self, dataset_path: str, chunk_size: int = 100000) -> Iterator[pd.DataFrame]:
        file_path = f"{dataset_path}/triage.csv"
        
        try:
            for df_chunk in pd.read_csv(file_path, chunksize=chunk_size):
                batch = []
                for _, row in df_chunk.iterrows():
                    text = row.get("text")
                    label = row.get("label")
                    
                    if pd.isna(text) or not str(text).strip():
                        continue
                        
                    triage_val = None
                    if label == "high":
                        triage_val = 2 # Emergent
                    elif label == "low":
                        triage_val = 4 # Less Urgent
                        
                    batch.append({
                        "dataset_source": self.dataset_source,
                        "raw_text": str(text).strip(),
                        "department": "Unknown",
                        "triage_level": triage_val,
                        "language": "en", # English text
                        "raw_severity": label
                    })
                    
                if batch:
                    yield pd.DataFrame(batch)
        except FileNotFoundError:
            pass
