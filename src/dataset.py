"""Dataset helpers for MediTriageAI."""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.model import SEVERITY_LABELS, SPECIALIST_CLASSES


class MediTriageDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer, max_length: int = 128):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]
        encoding = self.tokenizer(
            row["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels_specialist": torch.tensor(
                row["label_specialist_id"], dtype=torch.long
            ),
            "labels_severity": torch.tensor(row["label_severity_id"], dtype=torch.long),
            "id": row.get("id", str(idx)),
            "split": row.get("split", "unknown"),
            "dataset_source": row.get("dataset_source", "unknown"),
            "language": row.get("language", "unknown"),
        }


def load_split_rows(
    dataset_path: str | os.PathLike[str], split: str, max_rows: int | None = None
) -> list[dict]:
    path = str(dataset_path)

    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    from src.schema import validate_and_translate_schema

    df = validate_and_translate_schema(df)

    if "split" in df.columns:
        df_split = df[df["split"] == split].copy()
    else:
        # If no split column exists, assume the whole dataset is for this split (useful for small legacy test files)
        df_split = df.copy()

    if df_split["raw_text"].isna().sum() > 0:
        df_split = df_split.dropna(subset=["raw_text"])

    if max_rows is not None and max_rows > 0:
        from src.sampling import create_stratified_subset

        df_split = create_stratified_subset(
            df_split, max_rows, label_col="department", secondary_col="triage_level"
        )

    # Vectorized label mapping for high-performance loading across 10M rows
    spec_map = {cls_name: idx for idx, cls_name in enumerate(SPECIALIST_CLASSES)}
    sev_map = {lbl_name: idx for idx, lbl_name in enumerate(SEVERITY_LABELS)}

    dept_series = (
        df_split["department"].astype(object).map(spec_map).fillna(-1).astype(int)
    )
    triage_series = (
        df_split["triage_level"].astype(object).map(sev_map).fillna(-1).astype(int)
    )

    ids = (
        df_split["id"].astype(str)
        if "id" in df_split.columns
        else [f"sample_{i}" for i in range(len(df_split))]
    )
    splits = (
        df_split["split"].astype(str) if "split" in df_split.columns else split
    )
    sources = (
        df_split["dataset_source"].astype(str)
        if "dataset_source" in df_split.columns
        else "unknown"
    )
    languages = (
        df_split["language"].astype(str)
        if "language" in df_split.columns
        else "unknown"
    )

    df_out = pd.DataFrame(
        {
            "id": ids,
            "split": splits,
            "dataset_source": sources,
            "language": languages,
            "text": df_split["text"] if "text" in df_split.columns else df_split["raw_text"],
            "label_specialist_id": dept_series,
            "label_severity_id": triage_series,
        }
    )

    return df_out.to_dict("records")


class RunningMetrics:
    def __init__(self):
        self.loss_sum = 0.0
        self.specialist_loss_sum = 0.0
        self.severity_loss_sum = 0.0
        self.specialist_correct = 0
        self.severity_correct = 0
        self.total_samples = 0
        self.specialist_samples = 0
        self.severity_samples = 0

    def update(
        self,
        loss: float,
        specialist_loss: float,
        severity_loss: float,
        specialist_preds: list[int],
        specialist_labels: list[int],
        severity_preds: list[int],
        severity_labels: list[int],
    ):
        batch_size = len(specialist_labels)
        self.loss_sum += loss * batch_size
        self.total_samples += batch_size

        valid_spec_idx = [i for i, l in enumerate(specialist_labels) if l != -1]
        valid_sev_idx = [i for i, l in enumerate(severity_labels) if l != -1]

        self.specialist_samples += len(valid_spec_idx)
        self.severity_samples += len(valid_sev_idx)

        if len(valid_spec_idx) > 0:
            self.specialist_loss_sum += specialist_loss * len(valid_spec_idx)
            self.specialist_correct += sum(
                specialist_preds[i] == specialist_labels[i] for i in valid_spec_idx
            )

        if len(valid_sev_idx) > 0:
            self.severity_loss_sum += severity_loss * len(valid_sev_idx)
            self.severity_correct += sum(
                severity_preds[i] == severity_labels[i] for i in valid_sev_idx
            )

    def compute(self) -> dict[str, float]:
        if self.total_samples == 0:
            return {
                "loss": 0.0,
                "specialist_loss": 0.0,
                "severity_loss": 0.0,
                "specialist_acc": 0.0,
                "severity_acc": 0.0,
                "total_samples": 0,
            }

        return {
            "loss": self.loss_sum / self.total_samples,
            "specialist_loss": (
                self.specialist_loss_sum / self.specialist_samples
                if self.specialist_samples > 0
                else 0.0
            ),
            "severity_loss": (
                self.severity_loss_sum / self.severity_samples
                if self.severity_samples > 0
                else 0.0
            ),
            "specialist_acc": (
                self.specialist_correct / self.specialist_samples
                if self.specialist_samples > 0
                else 0.0
            ),
            "severity_acc": (
                self.severity_correct / self.severity_samples
                if self.severity_samples > 0
                else 0.0
            ),
            "total_samples": self.total_samples,
        }

    def reset(self):
        self.loss_sum = 0.0
        self.specialist_loss_sum = 0.0
        self.severity_loss_sum = 0.0
        self.specialist_correct = 0
        self.severity_correct = 0
        self.total_samples = 0
