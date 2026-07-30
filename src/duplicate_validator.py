import csv
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class DuplicateValidator:
    """Validate exact and near duplicates across datasets.

    Checks original and synthetic datasets for duplicate clinical notes.
    Generates a CSV report.
    """

    def __init__(self, original_path: Path, synthetic_path: Path):
        self.original_path = original_path
        self.synthetic_path = synthetic_path
        self.report_path = Path(
            "data/processed/enriched/duplicate_validation_report.csv"
        )
        self.report_path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self, path: Path) -> pd.DataFrame:
        return pd.read_csv(path)

    def _exact_duplicates(
        self, df1: pd.DataFrame, df2: pd.DataFrame, source1: str, source2: str
    ):
        dup_records = []
        set1 = set(df1["text"].fillna(""))
        for _, row in df2.iterrows():
            note = row.get("text", "")
            if note in set1:
                matched = df1[df1["text"] == note]
                dup_records.append(
                    {
                        "record_id": row.get("id", ""),
                        "duplicate_type": "exact",
                        "source_dataset": source2,
                        "matched_record_id": (
                            matched["id"].iloc[0] if "id" in matched.columns else ""
                        ),
                        "reason": f"Exact duplicate of {source1} record",
                    }
                )
        return dup_records

    def _near_duplicates(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        source1: str,
        source2: str,
        threshold: float = 0.9,
    ):
        dup_records = []

        def token_set(text: str):
            return set(text.lower().split())

        records1 = [
            (idx, token_set(row.get("text", ""))) for idx, row in df1.iterrows()
        ]
        for idx2, row2 in df2.iterrows():
            set2 = token_set(row2.get("text", ""))
            for idx1, set1 in records1:
                if not set1 or not set2:
                    continue
                intersect = len(set1 & set2)
                union = len(set1 | set2)
                if union == 0:
                    continue
                jaccard = intersect / union
                if threshold <= jaccard < 1.0:
                    dup_records.append(
                        {
                            "record_id": row2.get("id", ""),
                            "duplicate_type": "near",
                            "source_dataset": source2,
                            "matched_record_id": df1.iloc[idx1].get("id", ""),
                            "reason": f"Near duplicate (Jaccard {jaccard:.2f}) of {source1} record",
                        }
                    )
                    break
        return dup_records

    def validate(self):
        original = self._load(self.original_path)
        synthetic = self._load(self.synthetic_path)
        records = []
        records.extend(
            self._exact_duplicates(original, synthetic, "original", "synthetic")
        )
        records.extend(
            self._near_duplicates(original, synthetic, "original", "synthetic")
        )
        with self.report_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "record_id",
                    "duplicate_type",
                    "source_dataset",
                    "matched_record_id",
                    "reason",
                ],
            )
            writer.writeheader()
            for r in records:
                writer.writerow(r)
        logger.info(f"Duplicate validation report written to {self.report_path}")


if __name__ == "__main__":
    original_path = Path("data/processed/improved/dataset_improved.csv")
    synthetic_path = Path("data/processed/enriched/synthetic_samples.csv")
    validator = DuplicateValidator(original_path, synthetic_path)
    validator.validate()
