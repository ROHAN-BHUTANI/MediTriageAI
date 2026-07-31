import json
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from meditriage.builder.adapters.base import BaseAdapter


class KaggleMedicalTriageAdapter(BaseAdapter):
    """
    Adapter for Turkish Medical Emergency Triage dataset.

    Dataset:
        datasets/raw/kaggle_medical_triage/medical_data.json

    Mapping:
        input_text      -> complaint
        symptoms        -> symptom list
        urgency_level   -> triage level
        urgency_label   -> raw severity
        reasoning       -> reasoning
        response        -> recommendation
    """

    @property
    def dataset_source(self) -> str:
        return "kaggle_medical_triage"

    @property
    def version(self) -> str:
        return "2.0"

    def ingest(
        self,
        dataset_path: str,
        chunk_size: int = 100000,
    ) -> Iterator[pd.DataFrame]:

        json_path = Path(dataset_path) / "medical_data.json"

        if not json_path.exists():
            return

        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        batch = []

        for row in data:

            complaint = str(row.get("input_text", "")).strip()

            if not complaint:
                continue

            symptoms = row.get("symptoms", [])

            if isinstance(symptoms, list):
                symptoms = ", ".join(symptoms)

            raw_text = (
                f"Chief Complaint: {complaint}\n"
                f"Symptoms: {symptoms}\n"
                f"Clinical Reasoning: {row.get('reasoning','')}\n"
                f"Recommendation: {row.get('response','')}"
            )

            batch.append(
                {
                    "dataset_source": self.dataset_source,
                    "raw_text": raw_text,
                    "department": "ED",
                    "triage_level": row.get("urgency_level"),
                    "language": "tr",
                    "raw_severity": row.get("urgency_label"),
                }
            )

            if len(batch) >= chunk_size:
                yield pd.DataFrame(batch)
                batch = []

        if batch:
            yield pd.DataFrame(batch)
