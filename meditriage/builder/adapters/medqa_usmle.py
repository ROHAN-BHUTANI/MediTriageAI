from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .base import BaseAdapter


class MedqaUsmleAdapter(BaseAdapter):
    """
    Adapter for the MedQA USMLE dataset.

    Mapping Strategy:
    - `question` -> `raw_text`
    - Recursively reads any *.jsonl files found in raw directory.
    """

    @property
    def dataset_source(self) -> str:
        return "medqa_usmle"

    @property
    def version(self) -> str:
        return "1.0.0"

    def ingest(self, raw_path: str, chunk_size: int = 1000) -> Iterator[pd.DataFrame]:
        raw_dir = Path(raw_path)
        jsonl_files = list(raw_dir.rglob("*.jsonl"))

        if not jsonl_files:
            return

        for jsonl_path in jsonl_files:
            try:
                for chunk_idx, chunk_df in enumerate(
                    pd.read_json(jsonl_path, lines=True, chunksize=chunk_size)
                ):
                    records = []

                    for idx, row in chunk_df.iterrows():
                        text = str(row.get("question", "")).strip()
                        if not text or text.lower() == "nan":
                            continue

                        ans = str(row.get("answer", "")).lower()
                        opts = " ".join(str(v) for v in row.get("options", {}).values()).lower() if isinstance(row.get("options"), dict) else ""
                        comb = (text + " " + ans + " " + opts).lower()

                        department = "GEN_MED"
                        if any(k in comb for k in ["pediatric", "neonatal", "infant", "child", "boy", "girl", "newborn", "toddler"]):
                            department = "PEDS"
                        elif any(k in comb for k in ["gynecolog", "obstetric", "pregnancy", "pregnant", "delivered", "ovarian", "uterine", "placenta", "cervical cancer"]):
                            department = "OBGYN"
                        elif any(k in comb for k in ["oncolog", "carcinoma", "leukemia", "lymphoma", "metastatic", "chemotherapy", "melanoma", "sarcoma", "tumor", "cancer", "blastoma"]):
                            department = "ONCOLOGY_HEME"
                        elif any(k in comb for k in ["cardiolog", "pulmonolog", "cardiac", "heart", "coronary", "infarction", "arrhythmia", "pulmonary", "pneumonia", "embolism", "hypertension", "aortic"]):
                            department = "CARDIO_PULM"
                        elif any(k in comb for k in ["neurolog", "neurosurg", "stroke", "seizure", "epilepsy", "brain", "cerebral", "aneurysm", "encephalopathy", "dementia", "headache", "meningitis"]):
                            department = "NEURO"
                        elif any(k in comb for k in ["orthoped", "fracture", "dislocation", "femur", "tibia", "joint", "knee", "hip", "spine", "spinal cord", "scoliosis"]):
                            department = "ORTHO"
                        elif any(k in comb for k in ["gastroenterolog", "gastrointestinal", "colon", "liver", "hepatic", "pancreatic", "bowel", "gastric", "esophag", "appendicitis", "cirrhosis", "diarrhea"]):
                            department = "GI"
                        elif any(k in comb for k in ["urolog", "nephrolog", "kidney", "renal", "dialysis", "bladder", "prostate", "glomerulonephritis"]):
                            department = "RENAL_URO"
                        elif any(k in comb for k in ["ophthalmolog", "dermatolog", "otolaryngolog", "cornea", "retinal", "cataract", "glaucoma", "psoriasis", "eczema", "sinusitis", "tonsillitis"]):
                            department = "ENT_OPHTHALMO"
                        elif any(k in comb for k in ["psychiatr", "schizophrenia", "bipolar", "depression", "suicidal", "psychosis", "anorexia"]):
                            department = "PSYCH"
                        elif any(k in comb for k in ["surgeon", "surgery", "resection", "transplantation", "laparotom", "mastectomy", "graft", "postoperative"]):
                            department = "SURGERY"

                        records.append(
                            {
                                "tracking_id": f"medqa_usmle::{idx}::0",
                                "seed_id": f"medqa_usmle::{idx}",
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
                                "extraction_timestamp": datetime.now(timezone.utc).isoformat(),
                                "original_schema_version": self.version,
                            }
                        )

                    if records:
                        yield pd.DataFrame(records)
            except Exception:
                continue
