#!/usr/bin/env python3
"""
Aggregate MediTriageAI ablation experiment results.

This script:
- Discovers all completed experiments.
- Reads best_metrics.json from every experiment.
- Groups experiments by architecture.
- Computes summary statistics.
- Generates publication-ready CSV and Markdown reports.

Author: MediTriageAI Research Team
Project: MediTriageAI
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = ROOT / "experiments"
OUTPUT_DIR = ROOT / "analysis" / "results" / "ablation_summary"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def architecture_from_name(name: str) -> str:
    name = name.upper()

    if "FULL_ARCHITECTURE" in name:
        return "full_architecture"
    if "BASELINE" in name:
        return "baseline"
    if "ACES_ONLY" in name:
        return "aces_only"
    if "AMCO_ONLY" in name:
        return "amco_only"
    if "CCSM_ONLY" in name:
        return "ccsm_only"
    if "DCCF_ONLY" in name:
        return "dccf_only"

    return "unknown"


def seed_from_name(name: str):
    parts = name.upper().split("_")
    if "SEED" not in parts:
        return None
    idx = parts.index("SEED")
    if idx + 1 >= len(parts):
        return None
    try:
        return int(parts[idx + 1])
    except ValueError:
        return None


records = []

for metrics_file in sorted(EXPERIMENTS_DIR.glob("*/best_metrics.json")):
    exp_dir = metrics_file.parent

    with open(metrics_file, "r") as f:
        metrics = json.load(f)

    records.append(
        {
            "experiment": exp_dir.name,
            "architecture": architecture_from_name(exp_dir.name),
            "seed": seed_from_name(exp_dir.name),
            "epoch": metrics.get("epoch"),
            "training_time": metrics.get("time"),
            "train_loss": metrics.get("train_loss"),
            "train_specialist_acc": metrics.get("train_specialist_acc"),
            "train_severity_acc": metrics.get("train_severity_acc"),
            "val_loss": metrics.get("val_loss"),
            "val_specialist_acc": metrics.get("val_specialist_acc"),
            "val_severity_acc": metrics.get("val_severity_acc"),
        }
    )

df = pd.DataFrame(records)

print("=" * 70)
print("Experiments discovered :", len(df))
print(df["architecture"].value_counts().sort_index())
print("=" * 70)
