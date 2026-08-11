import json
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from meditriage.builder.adapters.base import BaseAdapter


class KaggleMedicalTriageAdapter(BaseAdapter):
    """
    Adapter for Medical Emergency Triage dataset.
    Supports medical_data.json, triage.csv, and parquet formats.
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

        raw_dir = Path(dataset_path)
        json_path = raw_dir / "medical_data.json"
        csv_path = raw_dir / "triage.csv"
        parquet_files = sorted(
            [f for f in raw_dir.rglob("*.parquet") if ".cache" not in str(f)]
        )

        data = []
        if json_path.exists():
            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        elif csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                data = df.to_dict(orient="records")
            except Exception:
                pass
        elif parquet_files:
            try:
                dfs = [pd.read_parquet(pf) for pf in parquet_files]
                df = pd.concat(dfs, ignore_index=True)
                data = df.to_dict(orient="records")
            except Exception:
                pass

        batch = []

        for row in data:
            complaint = str(row.get("input_text") or row.get("symptom_description") or row.get("text") or "").strip()

            if not complaint:
                continue

            symptoms = row.get("symptoms", [])

            if isinstance(symptoms, list):
                symptoms = ", ".join(symptoms)
            else:
                symptoms = str(symptoms or "")

            urgency_lvl = row.get("urgency_level") or row.get("label") or row.get("triage_level")
            dept = row.get("primary_specialty") or "ED"
            lang = "en" if "symptom_description" in row else "tr"

            raw_text = (
                f"Chief Complaint: {complaint}\n"
                f"Symptoms: {symptoms}\n"
                f"Clinical Reasoning: {row.get('reasoning', '')}\n"
                f"Recommendation: {row.get('response', '')}"
            )

            batch.append(
                {
                    "dataset_source": self.dataset_source,
                    "raw_text": raw_text,
                    "department": dept,
                    "triage_level": urgency_lvl,
                    "language": lang,
                    "raw_severity": row.get("urgency_label"),
                }
            )

            if len(batch) >= chunk_size:
                yield pd.DataFrame(batch)
                batch = []

        if batch:
            yield pd.DataFrame(batch)
