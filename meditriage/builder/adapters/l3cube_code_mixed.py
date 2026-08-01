import os
from collections.abc import Iterator

import pandas as pd

from meditriage.builder.adapters.base import BaseAdapter


class L3CubeCodeMixedAdapter(BaseAdapter):
    """
    Adapter for L3Cube-HingLID code-mixed dataset.
    Extracts Hinglish code-mixed sentences from token-level annotations.
    """

    @property
    def dataset_source(self) -> str:
        return "l3cube_code_mixed"

    @property
    def version(self) -> str:
        return "1.0"

    def ingest(
        self, dataset_path: str, chunk_size: int = 100000
    ) -> Iterator[pd.DataFrame]:
        base_dir = os.path.join(dataset_path, "code-mixed-nlp-main", "L3Cube-HingLID")
        files = ["train.txt", "validation.txt", "test.txt"]

        for file_name in files:
            file_path = os.path.join(base_dir, file_name)
            if not os.path.exists(file_path):
                # Fallback to the nested directory structure if unzipped differently
                file_path = os.path.join(
                    dataset_path,
                    "code-mixed-nlp",
                    "code-mixed-nlp-main",
                    "L3Cube-HingLID",
                    file_name,
                )
                if not os.path.exists(file_path):
                    continue

            with open(file_path, "r", encoding="utf-8") as f:
                batch = []
                current_sentence = []

                def map_hinglish_dept(text: str) -> str:
                    txt = text.lower()
                    if any(k in txt for k in ["baccha", "bacche", "baby", "toddler", "kid", "chota baccha"]):
                        return "PEDS"
                    elif any(k in txt for k in ["pregnant", "mahina", "period", "gynaec", "delivery", "garbh"]):
                        return "OBGYN"
                    elif any(k in txt for k in ["sar", "sirdard", "chakkar", "chhakkar", "headache", "seizure", "behoosh", "brain"]):
                        return "NEURO"
                    elif any(k in txt for k in ["chhati", "dil", "sans", "breath", "heart", "cough", "khansi", "bp", "fever", "bukhar"]):
                        return "CARDIO_PULM"
                    elif any(k in txt for k in ["haddi", "jod", "ghutna", "kamar", "spine", "joint", "knee", "bone", "leg", "haath", "pair"]):
                        return "ORTHO"
                    elif any(k in txt for k in ["pet", "petdard", "dast", "vomit", "ultee", "kabz", "gas", "stomach"]):
                        return "GI"
                    elif any(k in txt for k in ["peshab", "kidney", "urine"]):
                        return "RENAL_URO"
                    elif any(k in txt for k in ["aankh", "kaan", "gala", "throat", "skin", "khaj", "pimple", "eye", "ear"]):
                        return "ENT_OPHTHALMO"
                    elif any(k in txt for k in ["tension", "depress", "chinta", "stress", "mind"]):
                        return "PSYCH"
                    elif any(k in txt for k in ["cancer", "gath", "tumor"]):
                        return "ONCOLOGY_HEME"
                    elif any(k in txt for k in ["operation", "stitching", "surgery"]):
                        return "SURGERY"
                    return "GEN_MED"

                for line in f:
                    line = line.strip()
                    if not line:
                        if current_sentence:
                            sent_text = " ".join(current_sentence)
                            batch.append(
                                {
                                    "dataset_source": self.dataset_source,
                                    "raw_text": sent_text,
                                    "department": map_hinglish_dept(sent_text),
                                    "triage_level": None,
                                    "language": "hi-en",  # Hinglish
                                }
                            )
                            current_sentence = []

                            if len(batch) >= chunk_size:
                                yield pd.DataFrame(batch)
                                batch = []
                        continue

                    parts = line.split("\t")
                    if len(parts) >= 1:
                        current_sentence.append(parts[0])

                if current_sentence:
                    sent_text = " ".join(current_sentence)
                    batch.append(
                        {
                            "dataset_source": self.dataset_source,
                            "raw_text": sent_text,
                            "department": map_hinglish_dept(sent_text),
                            "triage_level": None,
                            "language": "hi-en",
                        }
                    )

                if batch:
                    yield pd.DataFrame(batch)
