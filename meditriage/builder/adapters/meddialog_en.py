import json
from collections.abc import Iterator
from pathlib import Path
import pandas as pd

from meditriage.builder.adapters.base import BaseAdapter


class MeddialogEnAdapter(BaseAdapter):
    """Adapter for MedDialog (English) dataset.

    Extracts dialogues between patients and doctors from json, jsonl, or parquet files.
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
        raw_dir = Path(dataset_path)
        
        # 1. Try Parquet files
        parquet_files = list(raw_dir.rglob("*.parquet"))
        if parquet_files:
            batch = []
            for pq_f in parquet_files:
                try:
                    df = pd.read_parquet(pq_f)
                    for _, row in df.iterrows():
                        utterances = row.get("utterances", [])
                        if isinstance(utterances, (list, tuple)):
                            raw_text = "\n".join(str(u) for u in utterances if u)
                        elif isinstance(utterances, str):
                            raw_text = utterances
                        else:
                            desc = str(row.get("description", ""))
                            raw_text = desc if desc else ""

                        if not raw_text.strip():
                            continue

                        comb_text = raw_text.lower()
                        department = self._classify_department(comb_text)

                        batch.append({
                            "dataset_source": self.dataset_source,
                            "raw_text": raw_text,
                            "department": department,
                            "triage_level": None,
                            "language": "en",
                        })

                        if len(batch) >= chunk_size:
                            yield pd.DataFrame(batch)
                            batch = []
                except Exception:
                    pass
            if batch:
                yield pd.DataFrame(batch)
            return

        # 2. Try JSON / JSONL files
        json_files = list(raw_dir.rglob("*.jsonl")) + list(raw_dir.rglob("*.json"))
        for j_f in json_files:
            if j_f.name == "SOURCE_URL.txt":
                continue
            batch = []
            try:
                with open(j_f, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            utterances = data.get("utterances", [])
                            if not utterances:
                                continue
                            raw_text = "\n".join(str(u) for u in utterances)
                            comb_text = raw_text.lower()
                            department = self._classify_department(comb_text)

                            batch.append({
                                "dataset_source": self.dataset_source,
                                "raw_text": raw_text,
                                "department": department,
                                "triage_level": None,
                                "language": "en",
                            })

                            if len(batch) >= chunk_size:
                                yield pd.DataFrame(batch)
                                batch = []
                        except Exception:
                            continue
            except Exception:
                pass
            if batch:
                yield pd.DataFrame(batch)

    def _classify_department(self, comb_text: str) -> str:
        if any(k in comb_text for k in ["pediatric", "neonatal", "infant", "child", "baby", "toddler"]):
            return "PEDS"
        elif any(k in comb_text for k in ["gynecolog", "obstetric", "pregnancy", "pregnant", "period", "menstrual", "vagina", "ovary", "uterus"]):
            return "OBGYN"
        elif any(k in comb_text for k in ["oncolog", "carcinoma", "leukemia", "lymphoma", "metastatic", "chemotherapy", "melanoma", "sarcoma", "tumor", "cancer"]):
            return "ONCOLOGY_HEME"
        elif any(k in comb_text for k in ["cardiolog", "pulmonolog", "cardiac", "heart", "hypertension", "chest pain", "asthma", "bronchitis", "pneumonia"]):
            return "CARDIO_PULM"
        elif any(k in comb_text for k in ["neurolog", "neurosurg", "stroke", "seizure", "epilepsy", "brain", "cerebral", "aneurysm", "headache"]):
            return "NEURO"
        elif any(k in comb_text for k in ["orthoped", "fracture", "sprain", "joint", "knee", "bone", "spine", "back pain", "arthritis"]):
            return "ORTHO"
        elif any(k in comb_text for k in ["gastroenterolog", "gastrointestinal", "colon", "liver", "hepatic", "pancreatic", "bowel", "gastric", "ulcer"]):
            return "GI"
        elif any(k in comb_text for k in ["urolog", "nephrolog", "kidney", "renal", "dialysis", "bladder", "prostate"]):
            return "RENAL_URO"
        elif any(k in comb_text for k in ["dermatolog", "ophthalmolog", "otolaryngolog", "eye", "ear", "throat", "skin", "rash", "acne"]):
            return "ENT_OPHTHALMO"
        elif any(k in comb_text for k in ["psychiatr", "psycholog", "anxiety", "depression", "panic", "bipolar", "mental health"]):
            return "PSYCH"
        elif any(k in comb_text for k in ["surgeon", "surgery", "operation", "post-operative"]):
            return "SURGERY"
        return "GEN_MED"
