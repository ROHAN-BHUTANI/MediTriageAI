import json
import os
import pandas as pd
from typing import Iterator
from meditriage.builder.adapters.base import BaseAdapter

class NhamcsEdAdapter(BaseAdapter):
    """
    Adapter for the NHAMCS ED datasets (2019, 2020, 2021).
    Reads fixed-width CDC format data and parses it according to the layout dictionary.
    """
    
    @property
    def dataset_source(self) -> str:
        return "nhamcs_ed"
        
    @property
    def version(self) -> str:
        return "1.0"
        
    def _load_dict(self):
        dict_path = os.path.join(os.path.dirname(__file__), "nhamcs_dict.json")
        with open(dict_path, "r") as f:
            return json.load(f)

    def ingest(self, dataset_path: str, chunk_size: int = 100000) -> Iterator[pd.DataFrame]:
        col_dicts = self._load_dict()
        
        for year in ["2019", "2020", "2021"]:
            year_path = os.path.join(dataset_path, f"ed{year}")
            if not os.path.exists(year_path):
                continue
                
            data_file = os.path.join(year_path, f"ed{year}")
            if not os.path.exists(data_file):
                continue
                
            cols = col_dicts.get(year)
            if not cols:
                continue
                
            with open(data_file, 'r', encoding='ascii', errors='ignore') as f:
                batch = []
                for line in f:
                    row = {}
                    for col in cols:
                        row[col['name']] = line[col['start']:col['start']+col['length']].strip()
                    
                    triage = row.get("IMMEDR", "")
                    if triage in ["1", "2", "3", "4", "5"]:
                        triage_val = int(triage)
                    elif triage in ["01", "02", "03", "04", "05"]:
                        triage_val = int(triage)
                    else:
                        triage_val = None
                        
                    rfv1 = row.get("RFV1", "")
                    rfv2 = row.get("RFV2", "")
                    rfv3 = row.get("RFV3", "")
                    age = row.get("AGE", "")
                    sex = row.get("SEX", "")
                    
                    sex_str = "Female" if sex == "1" else "Male" if sex == "2" else "Unknown"
                    
                    raw_text = f"Age: {age}, Sex: {sex_str}\n"
                    raw_text += f"Reason for Visit 1 (Code): {rfv1}\n"
                    if rfv2 and rfv2 != "-0009": raw_text += f"Reason for Visit 2 (Code): {rfv2}\n"
                    if rfv3 and rfv3 != "-0009": raw_text += f"Reason for Visit 3 (Code): {rfv3}\n"
                    
                    batch.append({
                        "dataset_source": self.dataset_source,
                        "raw_text": raw_text,
                        "department": "Emergency",
                        "triage_level": triage_val,
                        "language": "en" # NHAMCS is US data
                    })
                    
                    if len(batch) >= chunk_size:
                        yield pd.DataFrame(batch)
                        batch = []
                        
                if batch:
                    yield pd.DataFrame(batch)
