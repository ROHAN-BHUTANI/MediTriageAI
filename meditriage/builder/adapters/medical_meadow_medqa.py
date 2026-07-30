from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import ijson
import pandas as pd

from .base import BaseAdapter


class MedicalMeadowMedqaAdapter(BaseAdapter):
    """
    Adapter for the Medical Meadow MedQA dataset.

    Mapping Strategy:
    - `input` or `instruction` -> `raw_text`
    - Stream JSON objects iteratively using ijson to respect memory constraints.
    """

    @property
    def dataset_source(self) -> str:
        return "medical_meadow_medqa"

    @property
    def version(self) -> str:
        return "1.0.0"

    def ingest(self, raw_path: str, chunk_size: int = 1000) -> Iterator[pd.DataFrame]:
        json_path = Path(raw_path) / "medical_meadow_medqa.json"
        if not json_path.exists():
            return

        with open(json_path, "rb") as f:
            records = []
            idx = 0
            for item in ijson.items(f, "item"):
                text = str(item.get("input", "")).strip()
                if not text:
                    text = str(item.get("instruction", "")).strip()
                if not text:
                    continue

                records.append(
                    {
                        "tracking_id": f"medical_meadow_medqa::{idx}::0",
                        "seed_id": f"medical_meadow_medqa::{idx}",
                        "dataset_source": self.dataset_source,
                        "raw_text": text,
                        "raw_medical_specialty": None,
                        "raw_severity": None,
                        "language": "en",
                        "text": text,
                        "department": None,
                        "routing_confidence": "low",
                        "triage_level": None,
                        "severity_label_source": "native",
                        "is_perturbed": False,
                        "variant_index": 0,
                        "split": None,
                        "extraction_timestamp": datetime.now(timezone.utc).isoformat(),
                        "original_schema_version": self.version,
                    }
                )

                idx += 1

                if len(records) >= chunk_size:
                    yield pd.DataFrame(records)
                    records = []

            if records:
                yield pd.DataFrame(records)
