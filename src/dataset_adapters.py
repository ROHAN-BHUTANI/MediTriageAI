import hashlib
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class UnifiedPatientRecord:
    complaint_text: str
    source_dataset: str
    language: str = "en"
    patient_id: str | None = None
    demographics: dict[str, Any] | None = None
    specialist_label: str | None = None
    severity_label: int | None = None
    diagnosis: str | None = None
    split: str = "train"
    metadata: dict[str, Any] = field(default_factory=dict)


def assign_split(text: str) -> str:
    """Deterministically assign a split (80/10/10) based on text hash."""
    h = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
    r = h % 100
    if r < 80:
        return "train"
    if r < 90:
        return "val"
    return "test"


class DatasetAdapter:
    """Base class for all dataset adapters."""

    def __init__(self, data_path: str):
        self.data_path = Path(data_path)

    def iter_records(self) -> Iterator[UnifiedPatientRecord]:
        raise NotImplementedError


class NHAMCSAdapter(DatasetAdapter):
    """Adapter for NHAMCS Emergency Department dataset."""

    def iter_records(self) -> Iterator[UnifiedPatientRecord]:
        for file in self.data_path.rglob("*.csv"):
            for chunk in pd.read_csv(
                file, chunksize=10000, low_memory=False, on_bad_lines="skip"
            ):
                for _, row in chunk.iterrows():
                    complaint = None
                    if "REASON" in row:
                        complaint = row["REASON"]
                    elif "RFV1" in row:
                        complaint = row["RFV1"]
                    elif "text" in row:
                        complaint = row["text"]

                    if pd.isna(complaint) or not str(complaint).strip():
                        continue

                    acuity = row.get("IMMED", row.get("severity_heuristic", None))
                    try:
                        acuity = int(acuity) if pd.notna(acuity) else None
                    except (ValueError, TypeError):
                        acuity = None

                    text = str(complaint).strip()
                    # Assign split dynamically since NHAMCS usually lacks official splits
                    split = (
                        row.get("split")
                        if "split" in row and pd.notna(row.get("split"))
                        else assign_split(text)
                    )

                    yield UnifiedPatientRecord(
                        patient_id=str(row.get("RESID", "")),
                        complaint_text=text,
                        source_dataset="nhamcs_ed",
                        specialist_label="ED",
                        severity_label=acuity,
                        demographics={"age": row.get("AGE"), "sex": row.get("SEX")},
                        split=split,
                        metadata=row.to_dict(),
                    )


class ChatDoctorAdapter(DatasetAdapter):
    """Adapter for ChatDoctor (HealthCareMagic, iCliniq) & similar conversational formats."""

    def __init__(self, data_path: str, dataset_name: str = "chatdoctor"):
        super().__init__(data_path)
        self.dataset_name = dataset_name

    def iter_records(self) -> Iterator[UnifiedPatientRecord]:
        for file in self.data_path.rglob("*"):
            if file.suffix not in [".json", ".jsonl", ".parquet", ".csv"]:
                continue

            # Use directory structure for splits if available (e.g. train/ val/ test/)
            folder_split = None
            if "train" in file.parts:
                folder_split = "train"
            elif "val" in file.parts or "valid" in file.parts:
                folder_split = "val"
            elif "test" in file.parts:
                folder_split = "test"

            try:
                if file.suffix == ".parquet":
                    df_iter = [pd.read_parquet(file)]
                elif file.suffix == ".csv":
                    df_iter = pd.read_csv(file, chunksize=10000, on_bad_lines="skip")
                else:
                    lines = True if file.suffix == ".jsonl" else False
                    try:
                        df_iter = pd.read_json(file, lines=lines, chunksize=10000)
                    except ValueError:
                        df_iter = [pd.read_json(file)]
            except Exception:
                continue

            for df in df_iter:
                for _, row in df.iterrows():
                    complaint = row.get("input", row.get("text", ""))
                    if pd.isna(complaint) or not str(complaint).strip():
                        continue

                    text = str(complaint).strip()
                    # Check for explicit split column, then folder, then hash
                    split = folder_split
                    if "split" in row and pd.notna(row["split"]):
                        split = str(row["split"])
                    if not split:
                        split = assign_split(text)

                    yield UnifiedPatientRecord(
                        complaint_text=text,
                        source_dataset=self.dataset_name,
                        specialist_label="GEN_MED",  # Fallback taxonomy
                        split=split,
                        metadata=row.to_dict(),
                    )


class PMCPatientsAdapter(DatasetAdapter):
    """Adapter for PMC-Patients massive scale text dataset."""

    def iter_records(self) -> Iterator[UnifiedPatientRecord]:
        for file in self.data_path.rglob("*.csv"):
            for chunk in pd.read_csv(
                file, chunksize=5000, on_bad_lines="skip", low_memory=False
            ):
                for _, row in chunk.iterrows():
                    text_val = row.get("patient", row.get("text", ""))
                    if pd.isna(text_val) or not str(text_val).strip():
                        continue

                    text = str(text_val).strip()
                    split = (
                        row.get("split")
                        if "split" in row and pd.notna(row.get("split"))
                        else assign_split(text)
                    )

                    yield UnifiedPatientRecord(
                        patient_id=str(row.get("patient_uid", "")),
                        complaint_text=text,
                        source_dataset="pmc_patients",
                        split=split,
                        metadata=row.to_dict(),
                    )


class L3CubeAdapter(DatasetAdapter):
    """Adapter for line-by-line TXT datasets (L3Cube, MedDialog)."""

    def __init__(self, data_path: str, dataset_name: str = "l3cube_code_mixed"):
        super().__init__(data_path)
        self.dataset_name = dataset_name
        self.language = "en-hi" if "l3cube" in dataset_name else "en"

    def iter_records(self) -> Iterator[UnifiedPatientRecord]:
        for file in self.data_path.rglob("*.txt"):
            folder_split = None
            if "train" in file.parts or "train" in file.name.lower():
                folder_split = "train"
            elif "val" in file.parts or "val" in file.name.lower():
                folder_split = "val"
            elif "test" in file.parts or "test" in file.name.lower():
                folder_split = "test"

            with open(file, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    text = line.strip()
                    if not text:
                        continue

                    split = folder_split if folder_split else assign_split(text)

                    yield UnifiedPatientRecord(
                        complaint_text=text,
                        source_dataset=self.dataset_name,
                        language=self.language,
                        specialist_label="GEN_MED",
                        split=split,
                    )
