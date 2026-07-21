"""Auditing, validation, splitting, tokenization, and PyTorch dataloaders for E-PATH-CO-REASON."""

from __future__ import annotations

import os
import random
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from models.emergent_path_triage.exceptions import InterfaceError
from src.model import SEVERITY_LABELS, SPECIALIST_CLASSES

# Define standard regex-based language markers
EN_REGEX = re.compile(r"^[a-zA-Z0-9\s\.,!\?\(\)\'\":;-]+$")


@dataclass
class EmergentPathDataConfig:
    """Hyperparameter configurations for E-PATH-CO-REASON data preprocessing."""
    dataset_path: str = "meditriage/data/processed/dataset.csv"
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    random_seed: int = 1337
    max_length: int = 128
    batch_size: int = 32
    num_workers: int = 0
    pin_memory: bool = True
    stratify: bool = True
    colab_persistent_dir: str = "/content/drive/MyDrive/MediTriageAI"
    use_amp: bool = True


def set_global_seeds(seed: int) -> None:
    """Set global reproducibility seeds across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def detect_colab_environment() -> dict[str, Any]:
    """Detect presence of Google Colab environment and GPU details."""
    import sys
    is_colab = "google.colab" in sys.modules or os.path.exists("/content")
    has_gpu = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if has_gpu else "N/A"
    
    mixed_precision_available = False
    if has_gpu:
        major, minor = torch.cuda.get_device_capability(0)
        if major >= 7:
            mixed_precision_available = True
            
    def mount_google_drive(mount_point: str = "/content/drive") -> bool:
        if is_colab:
            from google.colab import drive
            drive.mount(mount_point)
            return True
        return False
        
    return {
        "is_colab": is_colab,
        "has_gpu": has_gpu,
        "gpu_name": gpu_name,
        "mixed_precision_available": mixed_precision_available,
        "mount_drive": mount_google_drive,
        "persistent_checkpoint_dir": "/content/drive/MyDrive/MediTriageAI" if is_colab else "./results"
    }


class LabelValidator:
    """Validator for specialist classes and severity labels."""
    
    def __init__(self) -> None:
        self.specialist_classes = SPECIALIST_CLASSES
        self.severity_labels = SEVERITY_LABELS
        
        self.spec_to_id = {c: i for i, c in enumerate(self.specialist_classes)}
        self.id_to_spec = {i: c for i, c in enumerate(self.specialist_classes)}
        
        self.sev_to_id = {l: i for i, l in enumerate(self.severity_labels)}
        self.id_to_sev = {i: l for i, l in enumerate(self.severity_labels)}

    def validate_specialist(self, label: str) -> int:
        """Validate specialist class existence and return index."""
        if label not in self.spec_to_id:
            raise ValueError(
                f"Unseen specialist class label: '{label}'. "
                f"Must be one of {self.specialist_classes}."
            )
        return self.spec_to_id[label]

    def validate_severity(self, label: str) -> int:
        """Validate severity label existence and return index."""
        if label not in self.sev_to_id:
            raise ValueError(
                f"Unseen severity label: '{label}'. "
                f"Must be one of {self.severity_labels}."
            )
        return self.sev_to_id[label]

    def serialize(self) -> dict[str, Any]:
        """Serialize label dictionary mappings."""
        return {
            "specialist_classes": self.specialist_classes,
            "severity_labels": self.severity_labels,
            "spec_to_id": self.spec_to_id,
            "sev_to_id": self.sev_to_id
        }


def get_leakage_safe_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 1337,
    stratify: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset grouped by seed_id to prevent clinical description leakage."""
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-5:
        raise ValueError("Train, validation, and test split ratios must sum to 1.0")
        
    df = df.copy()
    if "seed_id" not in df.columns:
        df["seed_id"] = df.index.map(str)

    seed_groups = df.groupby("seed_id")
    unique_seeds = list(seed_groups.groups.keys())
    
    seed_classes = []
    for sid in unique_seeds:
        grp = seed_groups.get_group(sid)
        cls_val = grp["department_code"].iloc[0] if "department_code" in grp.columns else "GEN_MED"
        seed_classes.append(cls_val)
        
    from sklearn.model_selection import train_test_split
    
    if stratify and len(set(seed_classes)) > 1:
        class_counts = pd.Series(seed_classes).value_counts()
        small_classes = set(class_counts[class_counts < 2].index)
        
        if len(small_classes) > 0:
            large_indices = [i for i, c in enumerate(seed_classes) if c not in small_classes]
            small_indices = [i for i, c in enumerate(seed_classes) if c in small_classes]
            
            large_seeds = [unique_seeds[i] for i in large_indices]
            large_classes = [seed_classes[i] for i in large_indices]
            small_seeds = [unique_seeds[i] for i in small_indices]
            
            if len(large_seeds) >= 2:
                train_seeds_l, temp_seeds_l = train_test_split(
                    large_seeds,
                    train_size=train_ratio,
                    random_state=seed,
                    stratify=large_classes
                )
                val_size_rel = val_ratio / (val_ratio + test_ratio)
                if len(temp_seeds_l) <= 1:
                    val_seeds_l, test_seeds_l = temp_seeds_l, []
                else:
                    val_classes = [large_classes[large_seeds.index(s)] for s in temp_seeds_l]
                    val_seeds_l, test_seeds_l = train_test_split(
                        temp_seeds_l,
                        train_size=val_size_rel,
                        random_state=seed,
                        stratify=val_classes if len(set(val_classes)) > 1 else None
                    )
            else:
                train_seeds_l, val_seeds_l, test_seeds_l = large_seeds, [], []
                
            rng = random.Random(seed)
            shuffled_small = list(small_seeds)
            rng.shuffle(shuffled_small)
            n_small = len(shuffled_small)
            n_train_s = round(n_small * train_ratio)
            n_val_s = round(n_small * val_ratio)
            
            train_seeds_s = shuffled_small[:n_train_s]
            val_seeds_s = shuffled_small[n_train_s:n_train_s+n_val_s]
            test_seeds_s = shuffled_small[n_train_s+n_val_s:]
            
            train_seeds = set(train_seeds_l) | set(train_seeds_s)
            val_seeds = set(val_seeds_l) | set(val_seeds_s)
            test_seeds = set(test_seeds_l) | set(test_seeds_s)
        else:
            train_seeds, temp_seeds = train_test_split(
                unique_seeds,
                train_size=train_ratio,
                random_state=seed,
                stratify=seed_classes
            )
            temp_classes = [seed_classes[unique_seeds.index(s)] for s in temp_seeds]
            val_size_rel = val_ratio / (val_ratio + test_ratio)
            if len(temp_seeds) <= 1:
                val_seeds, test_seeds = temp_seeds, []
            else:
                val_seeds, test_seeds = train_test_split(
                    temp_seeds,
                    train_size=val_size_rel,
                    random_state=seed,
                    stratify=temp_classes if len(set(temp_classes)) > 1 else None
                )
            train_seeds, val_seeds, test_seeds = set(train_seeds), set(val_seeds), set(test_seeds)
    else:
        rng = random.Random(seed)
        shuffled = list(unique_seeds)
        rng.shuffle(shuffled)
        n_seeds = len(shuffled)
        n_train = round(n_seeds * train_ratio)
        n_val = round(n_seeds * val_ratio)
        
        train_seeds = set(shuffled[:n_train])
        val_seeds = set(shuffled[n_train:n_train+n_val])
        test_seeds = set(shuffled[n_train+n_val:])
        
    train_df = df[df["seed_id"].isin(train_seeds)].copy()
    val_df = df[df["seed_id"].isin(val_seeds)].copy()
    test_df = df[df["seed_id"].isin(test_seeds)].copy()
    
    return train_df, val_df, test_df


class TokenizerPipeline:
    """Reusable tokenization pipeline wrapper."""
    
    def __init__(self, tokenizer: Any, max_length: int = 128) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, texts: list[str]) -> dict[str, torch.Tensor]:
        return self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )


class EmergentTriageDataset(Dataset):
    """Pytorch Dataset class mapping dual labels for E-PATH-CO-REASON."""
    
    def __init__(
        self,
        texts: list[str],
        specialist_labels: list[int],
        severity_labels: list[int],
        tokenizer_pipeline: TokenizerPipeline
    ) -> None:
        self.texts = texts
        self.specialist_labels = specialist_labels
        self.severity_labels = severity_labels
        self.tokenizer_pipeline = tokenizer_pipeline

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        encoding = self.tokenizer_pipeline([self.texts[idx]])
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels_specialist": torch.tensor(self.specialist_labels[idx], dtype=torch.long),
            "labels_severity": torch.tensor(self.severity_labels[idx], dtype=torch.long)
        }


class EmergentTriageCollator:
    """Batch collation utility."""
    
    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        input_ids = torch.stack([x["input_ids"] for x in batch])
        attention_mask = torch.stack([x["attention_mask"] for x in batch])
        labels_specialist = torch.stack([x["labels_specialist"] for x in batch])
        labels_severity = torch.stack([x["labels_severity"] for x in batch])
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels_specialist": labels_specialist,
            "labels_severity": labels_severity
        }


def get_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = True
) -> DataLoader:
    """Create a PyTorch DataLoader helper."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=EmergentTriageCollator()
    )


def audit_dataset(csv_path: str, tokenizer: Any = None) -> dict[str, Any]:
    """Audit dataset content, distributions, and label statistics."""
    df = pd.read_csv(csv_path)
    
    total_samples = len(df)
    missing_samples = df["text"].isna().sum()
    empty_texts = (df["text"].astype(str).str.strip() == "").sum()
    
    # Missing labels
    missing_spec = df["department_code"].isna().sum() if "department_code" in df.columns else 0
    missing_sev = df["severity_heuristic"].isna().sum() if "severity_heuristic" in df.columns else 0
    
    # Duplicates
    duplicate_samples = df["text"].duplicated().sum()
    
    # Check duplicate labels: same text but different class labels
    # We group by text and check unique label combinations
    dup_labels_count = 0
    if "department_code" in df.columns and "severity_heuristic" in df.columns:
        text_groups = df.groupby("text")
        for text, grp in text_groups:
            if len(grp["department_code"].unique()) > 1 or len(grp["severity_heuristic"].unique()) > 1:
                dup_labels_count += len(grp)
                
    # Validate Labels
    validator = LabelValidator()
    invalid_spec = 0
    if "department_code" in df.columns:
        invalid_spec = (~df["department_code"].isin(validator.specialist_classes)).sum()
        
    invalid_sev = 0
    if "severity_heuristic" in df.columns:
        invalid_sev = (~df["severity_heuristic"].isin(validator.severity_labels)).sum()

    # Class frequencies
    spec_freq = {}
    if "department_code" in df.columns:
        spec_freq = df["department_code"].value_counts().to_dict()
        
    sev_freq = {}
    if "severity_heuristic" in df.columns:
        sev_freq = df["severity_heuristic"].value_counts().to_dict()

    # Multilingual languages
    lang_freq = {}
    if "language" in df.columns:
        lang_freq = df["language"].value_counts().to_dict()
    else:
        # Predict based on char regex
        df["pred_lang"] = df["text"].map(
            lambda t: "en" if pd.notna(t) and EN_REGEX.match(str(t)) else "hinglish"
        )
        lang_freq = df["pred_lang"].value_counts().to_dict()

    # Sequence lengths
    df["word_count"] = df["text"].map(lambda t: len(str(t).split()) if pd.notna(t) else 0)
    lengths_w = df["word_count"].tolist()
    
    word_stats = {
        "max": int(np.max(lengths_w)),
        "mean": float(np.mean(lengths_w)),
        "median": float(np.median(lengths_w)),
        "p25": float(np.percentile(lengths_w, 25)),
        "p75": float(np.percentile(lengths_w, 75)),
        "p90": float(np.percentile(lengths_w, 90)),
        "p95": float(np.percentile(lengths_w, 95)),
        "p99": float(np.percentile(lengths_w, 99))
    }

    token_stats = {}
    if tokenizer is not None:
        token_lengths = []
        # Sample or complete tokenization depending on size to prevent out-of-memory
        sample_df = df.sample(min(2000, len(df)), random_state=42)
        for t in sample_df["text"]:
            if pd.isna(t):
                continue
            ids = tokenizer.encode(str(t), truncation=False)
            token_lengths.append(len(ids))
        token_stats = {
            "max": int(np.max(token_lengths)),
            "mean": float(np.mean(token_lengths)),
            "median": float(np.median(token_lengths)),
            "p25": float(np.percentile(token_lengths, 25)),
            "p75": float(np.percentile(token_lengths, 75)),
            "p90": float(np.percentile(token_lengths, 90)),
            "p95": float(np.percentile(token_lengths, 95)),
            "p99": float(np.percentile(token_lengths, 99))
        }

    return {
        "total_samples": total_samples,
        "missing_samples": int(missing_samples),
        "missing_specialist": int(missing_spec),
        "missing_severity": int(missing_sev),
        "empty_text": int(empty_texts),
        "duplicate_samples": int(duplicate_samples),
        "duplicate_labels_count": int(dup_labels_count),
        "invalid_specialist_labels": int(invalid_spec),
        "invalid_severity_labels": int(invalid_sev),
        "specialist_frequencies": spec_freq,
        "severity_frequencies": sev_freq,
        "language_frequencies": lang_freq,
        "word_stats": word_stats,
        "token_stats": token_stats
    }


def generate_dataset_report(audit_results: dict[str, Any], output_path: str) -> None:
    """Generate structured markdown dataset audit report."""
    total = audit_results["total_samples"]
    
    spec_rows = ""
    for k, v in audit_results["specialist_frequencies"].items():
        pct = (v / total) * 100
        spec_rows += f"| {k} | {v} | {pct:.2f}% |\n"
        
    sev_rows = ""
    for k, v in audit_results["severity_frequencies"].items():
        pct = (v / total) * 100
        sev_rows += f"| {k} | {v} | {pct:.2f}% |\n"
        
    lang_rows = ""
    for k, v in audit_results["language_frequencies"].items():
        pct = (v / total) * 100
        lang_rows += f"| {k} | {v} | {pct:.2f}% |\n"

    report = f"""# E-PATH-CO-REASON Dataset Audit Report

## 1. Dataset Overview & Size Metrics
- **Total Samples**: {audit_results["total_samples"]}
- **Missing Samples (NaN text)**: {audit_results["missing_samples"]}
- **Empty / Whitespace-only Texts**: {audit_results["empty_text"]}
- **Duplicate Text Instances**: {audit_results["duplicate_samples"]}
- **Label Mismatches (Same text, different labels)**: {audit_results["duplicate_labels_count"]}
- **Invalid Specialist Labels**: {audit_results["invalid_specialist_labels"]}
- **Invalid Severity Labels**: {audit_results["invalid_severity_labels"]}

---

## 2. Label & Language Distributions

### Specialist Classes
| Class Code | Samples | Percentage |
| :--- | :---: | :---: |
{spec_rows}

### Severity Levels
| Level | Samples | Percentage |
| :--- | :---: | :---: |
{sev_rows}

### Languages
| Code | Samples | Percentage |
| :--- | :---: | :---: |
{lang_rows}

---

## 3. Sequence Length Statistics

### Word Count Statistics
- **Maximum Length**: {audit_results["word_stats"]["max"]} words
- **Average Length**: {audit_results["word_stats"]["mean"]:.2f} words
- **Median Length**: {audit_results["word_stats"]["median"]:.1f} words
- **25th / 75th Percentile**: {audit_results["word_stats"]["p25"]:.1f} / {audit_results["word_stats"]["p75"]:.1f} words
- **90th / 95th / 99th Percentile**: {audit_results["word_stats"]["p90"]:.1f} / {audit_results["word_stats"]["p95"]:.1f} / {audit_results["word_stats"]["p99"]:.1f} words

"""
    if "max" in audit_results.get("token_stats", {}):
        report += f"""
### Token Count Statistics (Sampled)
- **Maximum Length**: {audit_results["token_stats"]["max"]} tokens
- **Average Length**: {audit_results["token_stats"]["mean"]:.2f} tokens
- **Median Length**: {audit_results["token_stats"]["median"]:.1f} tokens
- **25th / 75th Percentile**: {audit_results["token_stats"]["p25"]:.1f} / {audit_results["token_stats"]["p75"]:.1f} tokens
- **90th / 95th / 99th Percentile**: {audit_results["token_stats"]["p90"]:.1f} / {audit_results["token_stats"]["p95"]:.1f} / {audit_results["token_stats"]["p99"]:.1f} tokens
"""

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
