import json
from collections.abc import Iterator

import pandas as pd

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

    def ingest(
        self, dataset_path: str, chunk_size: int = 100000
    ) -> Iterator[pd.DataFrame]:
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
                    comb_text = raw_text.lower()

                    department = "GEN_MED"
                    if any(k in comb_text for k in ["pediatric", "neonatal", "infant", "child", "baby", "toddler"]):
                        department = "PEDS"
                    elif any(k in comb_text for k in ["gynecolog", "obstetric", "pregnancy", "pregnant", "period", "menstrual", "vagina", "ovary", "uterus"]):
                        department = "OBGYN"
                    elif any(k in comb_text for k in ["oncolog", "carcinoma", "leukemia", "lymphoma", "metastatic", "chemotherapy", "melanoma", "sarcoma", "tumor", "cancer"]):
                        department = "ONCOLOGY_HEME"
                    elif any(k in comb_text for k in ["cardiolog", "pulmonolog", "cardiac", "heart", "hypertension", "chest pain", "asthma", "bronchitis", "pneumonia"]):
                        department = "CARDIO_PULM"
                    elif any(k in comb_text for k in ["neurolog", "neurosurg", "stroke", "seizure", "epilepsy", "brain", "cerebral", "aneurysm", "headache"]):
                        department = "NEURO"
                    elif any(k in comb_text for k in ["orthoped", "fracture", "sprain", "joint", "knee", "bone", "spine", "back pain", "arthritis"]):
                        department = "ORTHO"
                    elif any(k in comb_text for k in ["gastroenterolog", "gastrointestinal", "colon", "liver", "hepatic", "pancreatic", "bowel", "gastric", "ulcer"]):
                        department = "GI"
                    elif any(k in comb_text for k in ["urolog", "nephrolog", "kidney", "renal", "dialysis", "bladder", "prostate"]):
                        department = "RENAL_URO"
                    elif any(k in comb_text for k in ["dermatolog", "ophthalmolog", "otolaryngolog", "eye", "ear", "throat", "skin", "rash", "acne"]):
                        department = "ENT_OPHTHALMO"
                    elif any(k in comb_text for k in ["psychiatr", "psycholog", "anxiety", "depression", "panic", "bipolar", "mental health"]):
                        department = "PSYCH"
                    elif any(k in comb_text for k in ["surgeon", "surgery", "operation", "post-operative"]):
                        department = "SURGERY"

                    batch.append(
                        {
                            "dataset_source": self.dataset_source,
                            "raw_text": raw_text,
                            "department": department,
                            "triage_level": None,
                            "language": "en",
                        }
                    )

                    if len(batch) >= chunk_size:
                        yield pd.DataFrame(batch)
                        batch = []

                if batch:
                    yield pd.DataFrame(batch)
        except FileNotFoundError:
            pass
