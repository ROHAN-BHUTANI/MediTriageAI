"""Publication-Grade Experiment and Benchmark Report Generator."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from meditriage.training.config import TrainingConfig
from meditriage.training.utils import get_git_commit_hash, get_hardware_info

logger = logging.getLogger("meditriage.training")


def generate_experiment_reports(
    config: TrainingConfig,
    metrics: dict[str, Any],
    history: list[dict[str, Any]] | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Generate comprehensive publication-grade experiment and benchmark reports.

    Args:
        config: TrainingConfig instance.
        metrics: Final evaluated test/validation metrics.
        history: Training history log list.
        output_dir: Target report output directory.

    Returns:
        Master report dictionary.
    """
    out_dir = Path(output_dir or config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hardware_info = get_hardware_info()
    git_commit = get_git_commit_hash()

    master_report: dict[str, Any] = {}

    # 1. Training Summary JSON
    summary_data = {
        "experiment_name": config.experiment_name,
        "model_name": config.model_name_or_path,
        "epochs": config.num_epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "metrics": metrics,
    }
    master_report["summary"] = summary_data

    # 2. Benchmark Results JSON
    benchmark_data = {
        "model": config.model_name_or_path,
        "accuracy": metrics.get("test_accuracy", metrics.get("eval_accuracy", 0.0)),
        "balanced_accuracy": metrics.get(
            "test_balanced_accuracy", metrics.get("eval_balanced_accuracy", 0.0)
        ),
        "macro_f1": metrics.get("test_macro_f1", metrics.get("eval_macro_f1", 0.0)),
        "weighted_f1": metrics.get(
            "test_weighted_f1", metrics.get("eval_weighted_f1", 0.0)
        ),
        "top2_accuracy": metrics.get(
            "test_top2_accuracy", metrics.get("eval_top2_accuracy", 0.0)
        ),
        "calibration_error": metrics.get(
            "test_calibration_error", metrics.get("eval_calibration_error", 0.0)
        ),
    }
    master_report["benchmark"] = benchmark_data

    # 3. Hardware Report JSON
    master_report["hardware"] = hardware_info

    # 4. Reproducibility Report JSON
    reproducibility_data = {
        "git_commit": git_commit,
        "seed": config.seed,
        "config_snapshot": config.experiment_name,
        "environment": hardware_info,
    }
    master_report["reproducibility"] = reproducibility_data

    # Save JSON files
    with open(out_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    with open(out_dir / "benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=2)

    with open(out_dir / "hardware_report.json", "w", encoding="utf-8") as f:
        json.dump(hardware_info, f, indent=2)

    with open(out_dir / "reproducibility_report.json", "w", encoding="utf-8") as f:
        json.dump(reproducibility_data, f, indent=2)

    # 5. Experiment Report Markdown
    md_content = f"""# MediTriageAI Experiment Report: {config.experiment_name}

## Executive Summary
- **Model Backbone**: `{config.model_name_or_path}`
- **Primary Macro F1**: **{benchmark_data["macro_f1"]:.4f}**
- **Balanced Accuracy**: **{benchmark_data["balanced_accuracy"]:.4f}**
- **Git Commit**: `{git_commit}`

## Benchmark Performance
| Metric | Score |
| :--- | :--- |
| **Accuracy** | {benchmark_data["accuracy"]:.4f} |
| **Balanced Accuracy** | {benchmark_data["balanced_accuracy"]:.4f} |
| **Macro F1** | {benchmark_data["macro_f1"]:.4f} |
| **Weighted F1** | {benchmark_data["weighted_f1"]:.4f} |
| **Top-2 Accuracy** | {benchmark_data["top2_accuracy"]:.4f} |
| **Calibration Error (ECE)** | {benchmark_data["calibration_error"]:.4f} |

## Hardware & Environment
- **PyTorch Version**: `{hardware_info["torch_version"]}`
- **CUDA Device**: `{hardware_info["gpu_model"]}` ({hardware_info["gpu_count"]} GPU(s))
- **OS Platform**: `{hardware_info["os_platform"]}`
"""
    with open(out_dir / "experiment_report.md", "w", encoding="utf-8") as f:
        f.write(md_content)

    logger.info("Experiment reports generated successfully in %s", out_dir)
    return master_report
