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

                for line in f:
                    line = line.strip()
                    if not line:
                        if current_sentence:
                            batch.append(
                                {
                                    "dataset_source": self.dataset_source,
                                    "raw_text": " ".join(current_sentence),
                                    "department": None,
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
                    batch.append(
                        {
                            "dataset_source": self.dataset_source,
                            "raw_text": " ".join(current_sentence),
                            "department": None,
                            "triage_level": None,
                            "language": "hi-en",
                        }
                    )

                if batch:
                    yield pd.DataFrame(batch)
