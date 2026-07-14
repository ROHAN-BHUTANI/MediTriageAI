"""Dataset helpers for MediTriageAI."""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.model import SEVERITY_LABELS, SPECIALIST_CLASSES


class MediTriageDataset(Dataset):
    def __init__(self, rows: List[Dict[str, Any]], tokenizer, max_length: int = 128):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]
        encoding = self.tokenizer(
            row["text"], truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels_specialist": torch.tensor(row["label_specialist_id"], dtype=torch.long),
            "labels_severity": torch.tensor(row["label_severity_id"], dtype=torch.long),
        }


def load_split_rows(dataset_csv: str | "os.PathLike[str]", split: str, max_rows: int | None = None) -> list[dict]:
    df = pd.read_csv(dataset_csv)
    df_split = df[df["split"] == split].copy()
    if df_split["text"].isna().sum() > 0:
        df_split = df_split.dropna(subset=["text"])
    if "department_code" not in df_split.columns:
        raise KeyError("Processed dataset is missing 'department_code'.")
        
    severity_source = "severity_heuristic" if "severity_heuristic" in df_split.columns else "severity_label"
    
    if max_rows is not None and max_rows > 0:
        from src.sampling import create_stratified_subset
        df_split = create_stratified_subset(df_split, max_rows, label_col="department_code", secondary_col=severity_source)

    rows = []
    for _, row in df_split.iterrows():
        rows.append(
            {
                "text": row["text"],
                "label_specialist_id": SPECIALIST_CLASSES.index(str(row["department_code"])),
                "label_severity_id": SEVERITY_LABELS.index(str(row[severity_source])),
            }
        )
    return rows


class RunningMetrics:
    def __init__(self):
        self.loss_sum = 0.0
        self.specialist_loss_sum = 0.0
        self.severity_loss_sum = 0.0
        self.specialist_correct = 0
        self.severity_correct = 0
        self.total_samples = 0

    def update(self, loss: float, specialist_loss: float, severity_loss: float, specialist_preds: list[int], specialist_labels: list[int], severity_preds: list[int], severity_labels: list[int]):
        batch_size = len(specialist_labels)
        self.loss_sum += loss * batch_size
        self.specialist_loss_sum += specialist_loss * batch_size
        self.severity_loss_sum += severity_loss * batch_size
        self.specialist_correct += sum(p == l for p, l in zip(specialist_preds, specialist_labels))
        self.severity_correct += sum(p == l for p, l in zip(severity_preds, severity_labels))
        self.total_samples += batch_size

    def compute(self) -> dict[str, float]:
        if self.total_samples == 0:
            return {"loss": 0.0, "specialist_loss": 0.0, "severity_loss": 0.0, "specialist_acc": 0.0, "severity_acc": 0.0, "total_samples": 0}
        return {
            "loss": self.loss_sum / self.total_samples,
            "specialist_loss": self.specialist_loss_sum / self.total_samples,
            "severity_loss": self.severity_loss_sum / self.total_samples,
            "specialist_acc": self.specialist_correct / self.total_samples,
            "severity_acc": self.severity_correct / self.total_samples,
            "total_samples": self.total_samples,
        }

    def reset(self):
        self.loss_sum = 0.0
        self.specialist_loss_sum = 0.0
        self.severity_loss_sum = 0.0
        self.specialist_correct = 0
        self.severity_correct = 0
        self.total_samples = 0
