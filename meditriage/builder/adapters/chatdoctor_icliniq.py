from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from .base import BaseAdapter


class ChatDoctorIcliniqAdapter(BaseAdapter):
    """
    Adapter for the ChatDoctor iCliniq dataset.

    Mapping Strategy:
    - `input` -> `raw_text`
    - Stream parquet using pyarrow.
    """

    @property
    def dataset_source(self) -> str:
        return "chatdoctor_icliniq"

    @property
    def version(self) -> str:
        return "1.0.0"

    def ingest(self, raw_path: str, chunk_size: int = 1000) -> Iterator[pd.DataFrame]:
        data_dir = Path(raw_path) / "data"
        if not data_dir.exists():
            return

        parquet_files = list(data_dir.glob("*.parquet"))
        if not parquet_files:
            return

        for pfile in parquet_files:
            parquet_file = pq.ParquetFile(pfile)

            for batch in parquet_file.iter_batches(batch_size=chunk_size):
                chunk_df = batch.to_pandas()
                records = []

                for idx, row in chunk_df.iterrows():
                    # Clean and extract
                    text = str(row.get("input", "")).strip()
                    if not text or text.lower() == "nan":
                        continue

                    # Deterministic specialty routing (prefer doctor answer over patient query)
                    doctor_ans = str(row.get("answer_icliniq") or row.get("answer_chatdoctor") or "").strip().lower()
                    patient_q = text.lower()
                    comb_text = doctor_ans + " " + patient_q

                    department = "GEN_MED"

                    peds_m = "pediatric" in doctor_ans or "pediatrician" in doctor_ans or "baby" in doctor_ans or "infant" in doctor_ans or "toddler" in doctor_ans or "child" in doctor_ans
                    peds_p = "pediatric" in patient_q or "pediatrician" in patient_q or "baby" in patient_q or "infant" in patient_q or "toddler" in patient_q or "child" in patient_q

                    if peds_m or peds_p:
                        department = "PEDS"
                    elif any(k in comb_text for k in ["gynecolog", "obstetric", "pregnancy", "pregnant", "period", "menstrual", "vagina", "ovary", "uterus", "pap smear"]):
                        department = "OBGYN"
                    elif any(k in comb_text for k in ["neurolog", "neurosurg", "vertigo", "seizure", "epilepsy", "migraine", "bppv", "numbness", "paralysis", "stroke"]):
                        department = "NEURO"
                    elif any(k in comb_text for k in ["cardiolog", "pulmonolog", "cardiac", "heart", "hypertension", "chest pain", "asthma", "bronchitis", "pneumonia", "shortness of breath"]):
                        department = "CARDIO_PULM"
                    elif any(k in comb_text for k in ["orthoped", "fracture", "sprain", "joint", "knee", "bone", "spine", "back pain", "arthritis", "ligament", "tendon"]):
                        department = "ORTHO"
                    elif any(k in comb_text for k in ["gastroenterolog", "gastrointestinal", "acidity", "gerd", "diarrhea", "constipation", "liver", "gallbladder", "appendix", "ulcer", "bowel"]):
                        department = "GI"
                    elif any(k in comb_text for k in ["urolog", "nephrolog", "kidney", "urinary", "uti", "prostate", "bladder", "testes", "testicle", "scrotum"]):
                        department = "RENAL_URO"
                    elif any(k in comb_text for k in ["dermatolog", "ophthalmolog", "otolaryngolog", "eye", "ear", "throat", "skin", "rash", "acne", "sinus", "tonsil", "vision"]):
                        department = "ENT_OPHTHALMO"
                    elif any(k in comb_text for k in ["psychiatr", "psycholog", "anxiety", "depression", "panic", "bipolar", "mental health"]):
                        department = "PSYCH"
                    elif any(k in comb_text for k in ["oncolog", "hematolog", "cancer", "tumor", "biopsy", "chemotherapy", "leukemia", "lymphoma"]):
                        department = "ONCOLOGY_HEME"
                    elif any(k in comb_text for k in ["surgeon", "surgery", "operation", "post-operative", "incision"]):
                        department = "SURGERY"

                    # Build record
                    records.append(
                        {
                            "tracking_id": f"chatdoctor_icliniq::{pfile.name}::{idx}::0",
                            "seed_id": f"chatdoctor_icliniq::{pfile.name}::{idx}",
                            "dataset_source": self.dataset_source,
                            "raw_text": text,
                            "raw_medical_specialty": None,
                            "raw_severity": None,
                            "language": "en",
                            "text": text,
                            "department": department,
                            "routing_confidence": "high",
                            "triage_level": None,
                            "severity_label_source": "native",
                            "is_perturbed": False,
                            "variant_index": 0,
                            "split": None,
                            "extraction_timestamp": datetime.now(
                                timezone.utc
                            ).isoformat(),
                            "original_schema_version": self.version,
                        }
                    )

                if records:
                    yield pd.DataFrame(records)
