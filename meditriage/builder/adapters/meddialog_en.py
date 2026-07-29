import json
import pandas as pd
from typing import Iterator
from meditriage.builder.adapters.base import BaseAdapter

class MeddialogEnAdapter(BaseAdapter):
    """
    Adapter for MedDialog (English) dataset.
    Extracts dialogues between patients and doctors.
    """
    
    @property
    def dataset_source(self) -> str:
        return "meddialog_en"
        
    @property
    def version(self) -> str:
        return "1.0"
        
    def ingest(self, dataset_path: str, chunk_size: int = 100000) -> Iterator[pd.DataFrame]:
        file_path = f"{dataset_path}/dialog.jsonl"
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                batch = []
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                        
                    data = json.loads(line)
                    utterances = data.get("utterances", [])
                    
                    if not utterances:
                        continue
                        
                    raw_text = "\n".join(utterances)
                    
                    batch.append({
                        "dataset_source": self.dataset_source,
                        "raw_text": raw_text,
                        "department": "Unknown",
                        "triage_level": None,
                        "language": "en"
                    })
                    
                    if len(batch) >= chunk_size:
                        yield pd.DataFrame(batch)
                        batch = []
                        
                if batch:
                    yield pd.DataFrame(batch)
        except FileNotFoundError:
            pass
