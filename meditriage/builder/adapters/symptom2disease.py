from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from .base import BaseAdapter


class Symptom2DiseaseAdapter(BaseAdapter):
    """
    Adapter for the Symptom2Disease dataset.

    Mapping Strategy:
    - `text` -> `raw_text`
    - `label` -> `raw_medical_specialty`
    """

    @property
    def dataset_source(self) -> str:
        return "symptom2disease"

    @property
    def version(self) -> str:
        return "1.0.0"

    def ingest(self, raw_path: str, chunk_size: int = 1000) -> Iterator[pd.DataFrame]:
        csv_path = Path(raw_path) / "Symptom2Disease.csv"
        if not csv_path.exists():
            return

        for chunk_idx, chunk_df in enumerate(
            pd.read_csv(csv_path, chunksize=chunk_size)
        ):
            records = []

            for idx, row in chunk_df.iterrows():
                # Clean and extract
                text = str(row.get("text", "")).strip()
                if not text or text.lower() == "nan":
                    continue

                label = str(row.get("label", "")).strip()
                if label.lower() == "nan":
                    label = None

                # Map label to department
                dept_mapping = {
                    "Psoriasis": "ENT_OPHTHALMO",
                    "Varicose Veins": "CARDIO_PULM",
                    "Typhoid": "GEN_MED",
                    "Chicken pox": "GEN_MED",
                    "Impetigo": "ENT_OPHTHALMO",
                    "Dengue": "GEN_MED",
                    "Fungal infection": "ENT_OPHTHALMO",
                    "Common Cold": "GEN_MED",
                    "Pneumonia": "CARDIO_PULM",
                    "Dimorphic Hemorrhoids": "GI",
                    "Arthritis": "ORTHO",
                    "Acne": "ENT_OPHTHALMO",
                    "Bronchial Asthma": "CARDIO_PULM",
                    "Hypertension": "CARDIO_PULM",
                    "Migraine": "NEURO",
                    "Cervical spondylosis": "ORTHO",
                    "Jaundice": "GI",
                    "Malaria": "GEN_MED",
                    "urinary tract infection": "RENAL_URO",
                    "allergy": "ENT_OPHTHALMO",
                }
                department = dept_mapping.get(label, "GEN_MED") if label else None

                # Build record
                records.append(
                    {
                        "tracking_id": f"symptom2disease::{idx}::0",
                        "seed_id": f"symptom2disease::{idx}",
                        "dataset_source": self.dataset_source,
                        "raw_text": text,
                        "language": "en",
                        "department": department,
                        "triage_level": None,
                    }
                )

            if records:
                yield pd.DataFrame(records)
