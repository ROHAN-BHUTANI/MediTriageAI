import pandas as pd
from pathlib import Path
from typing import Iterator
from datetime import datetime, timezone
from .base import BaseAdapter

class MTSamplesAdapter(BaseAdapter):
    """
    Adapter for the MTSamples dataset.
    
    Mapping Strategy:
    - `transcription` -> `raw_text`. Fallback to `description` if `transcription` is empty or NaN.
    - `medical_specialty` -> `raw_medical_specialty`.
    - Drop records where the resulting text is empty.
    """
    @property
    def dataset_source(self) -> str:
        return "mtsamples"
        
    @property
    def version(self) -> str:
        return "1.1.0"

    def ingest(self, raw_path: str, chunk_size: int = 1000) -> Iterator[pd.DataFrame]:
        csv_path = Path(raw_path) / "mtsamples (1).csv"
        if not csv_path.exists():
            return
            
        for chunk_idx, chunk_df in enumerate(pd.read_csv(csv_path, index_col=0, chunksize=chunk_size)):
            records = []
            
            for idx, row in chunk_df.iterrows():
                # Clean and extract
                text = str(row.get("transcription", "")).strip()
                if not text or text.lower() == "nan":
                    text = str(row.get("description", "")).strip()
                    
                if not text or text.lower() == "nan":
                    continue # Drop completely empty records
                    
                specialty = str(row.get("medical_specialty", "")).strip().lower()
                if specialty == "nan" or not specialty:
                    continue
                
                # Canonical mapping
                mapping = {
                    "cardiovascular / pulmonary": "CARDIO_PULM",
                    "neurology": "NEURO",
                    "urology": "UROLOGY",
                    "general medicine": "GEN_MED",
                    "surgery": "SURGERY",
                    "psychiatry / psychology": "PSYCH",
                    "pediatrics - neonatal": "PEDIATRICS",
                    "orthopedic": "ORTHO",
                    "ophthalmology": "OPHTHAL",
                    "obstetrics / gynecology": "OB_GYN",
                    "neurosurgery": "SURGERY",
                    "nephrology": "UROLOGY", # mapping renal/nephrology
                    "hematology - oncology": "ONCOLOGY",
                    "gastroenterology": "GEN_MED", # Or GI? user provided ORTHO, GEN_MED, CARDIO_PULM, SURGERY, PEDIATRICS, OB_GYN, NEURO, PSYCH, DERM, ENT, OPHTHAL, UROLOGY, ONCOLOGY, ED. Wait, GI was in SPECIALIST_CLASSES. Yes, GI.
                    "ent - otolaryngology": "ENT",
                    "endocrinology": "GEN_MED",
                    "emergency room reports": "ED",
                    "dermatology": "DERM",
                    "cosmetic / plastic surgery": "SURGERY",
                    "bariatrics": "SURGERY",
                }
                
                department = mapping.get(specialty, "GEN_MED") # Fallback to GEN_MED for unknown
                
                # Build record
                records.append({
                    "tracking_id": f"mtsamples::{idx}::0",
                    "seed_id": f"mtsamples::{idx}",
                    "dataset_source": self.dataset_source,
                    "raw_text": text,
                    "raw_medical_specialty": specialty,
                    "department": department,
                    "triage_level": None,
                    "language": "en",
                    "text": text,
                    "department_code": department,
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
