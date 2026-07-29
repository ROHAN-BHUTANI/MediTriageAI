"""
Data Ingestion Layer for concrete training execution.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit


class DatasetNotFoundError(Exception):
    """Raised when required physical datasets are missing."""

class SchemaValidationError(Exception):
    """Raised when the dataset structure does not match expectations."""


class TriageDataset(Dataset):
    """PyTorch Dataset mapping for the E-PATH-CO-REASON structure."""
    
    def __init__(self, df: pd.DataFrame, tokenizer: Any, max_length: int = 256):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self) -> int:
        return len(self.df)
        
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        text = str(row['text'])
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels_specialist": torch.tensor(row['specialist_label'], dtype=torch.long),
            "labels_severity": torch.tensor(row['severity_label'], dtype=torch.long)
        }


def _validate_schema(df: pd.DataFrame) -> None:
    required_cols = {"patient_id", "text", "specialist_label", "severity_label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise SchemaValidationError(f"Missing required columns: {missing}")

    # Validate types
    if not pd.api.types.is_numeric_dtype(df['specialist_label']):
        raise SchemaValidationError("'specialist_label' must be numeric.")
    if not pd.api.types.is_numeric_dtype(df['severity_label']):
        raise SchemaValidationError("'severity_label' must be numeric.")


def load_dataset_file(filepath: str | Path) -> pd.DataFrame:
    path = Path(filepath)
    if not path.exists():
        raise DatasetNotFoundError(
            f"Required dataset file missing: {path}. "
            "Fabrication is strictly prohibited by Mission 07E."
        )
    
    df = pd.read_csv(path)
    _validate_schema(df)
    return df


def load_and_split_dataset(
    filepath: str | Path, 
    tokenizer: Any, 
    batch_size: int = 16,
    val_split: float = 0.1,
    test_split: float = 0.1,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Loads a dataset and splits it, strictly isolating patients."""
    df = load_dataset_file(filepath)
    
    # 1. Split out test set (patient-level)
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_split, random_state=seed)
    train_val_idx, test_idx = next(gss_test.split(df, groups=df['patient_id']))
    df_train_val = df.iloc[train_val_idx]
    df_test = df.iloc[test_idx]
    
    # 2. Split out validation set from the remaining (patient-level)
    # Adjust val_split proportion relative to the remaining data
    relative_val_split = val_split / (1.0 - test_split)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=relative_val_split, random_state=seed)
    train_idx, val_idx = next(gss_val.split(df_train_val, groups=df_train_val['patient_id']))
    
    df_train = df_train_val.iloc[train_idx]
    df_val = df_train_val.iloc[val_idx]
    
    train_dataset = TriageDataset(df_train, tokenizer)
    val_dataset = TriageDataset(df_val, tokenizer)
    test_dataset = TriageDataset(df_test, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def load_ood_dataset(
    filepath: str | Path, 
    tokenizer: Any, 
    batch_size: int = 16
) -> DataLoader:
    """Loads out-of-distribution dataset."""
    df = load_dataset_file(filepath)
    dataset = TriageDataset(df, tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

def load_hinglish_dataset(
    filepath: str | Path, 
    tokenizer: Any, 
    batch_size: int = 16
) -> DataLoader:
    """Loads Hinglish perturbation dataset."""
    df = load_dataset_file(filepath)
    dataset = TriageDataset(df, tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)
