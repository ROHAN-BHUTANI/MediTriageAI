"""Evaluation helpers for MediTriageAI."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:  # pragma: no cover
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

from src.evaluation import EvaluationExporter
from src.metrics import (
    classification_report,
    compute_macro_f1,
    compute_ordinal_confusion,
    compute_per_class_f1,
)

RESULTS_DIR = REPO_ROOT / "results"


def now_utc() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _tensor_like_list(values: Any) -> list[int]:
    if hasattr(values, "detach"):
        return values.detach().cpu().tolist()
    if hasattr(values, "cpu") and hasattr(values, "tolist"):
        return values.cpu().tolist()
    return list(values)


def run_evaluation(
    model: Any,
    tokenizer: Any,
    test_loader: Any,
    config: Any,
    expected_test_rows: int | None = None,
) -> dict[str, Any]:
    import torch

    model.eval()
    specialist_true: list[int] = []
    specialist_pred: list[int] = []
    severity_true: list[int] = []
    severity_pred: list[int] = []

    device = next(model.parameters()).device

    model_short_name = getattr(
        config, "model_short_name", getattr(model, "short_name", "unknown")
    )
    result_dir = RESULTS_DIR / model_short_name
    exporter = EvaluationExporter(str(result_dir))

    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)

        # Handle both ModelOutputs dataclass (EmergentPathTriageModel)
        # and legacy tuple returns (baseline models).
        # Verified: EmergentPathTriageModel.forward() returns ModelOutputs
        # with .specialist_logits and .severity_logits attributes.
        if hasattr(outputs, "specialist_logits") and hasattr(outputs, "severity_logits"):
            specialist_logits = outputs.specialist_logits
            severity_logits = outputs.severity_logits
        elif isinstance(outputs, tuple) and len(outputs) == 2:
            first, second = outputs
            # Convention: severity has 5 classes, specialist has 13
            if first.shape[-1] == 5:
                severity_logits, specialist_logits = first, second
            else:
                specialist_logits, severity_logits = first, second
        else:
            raise TypeError(
                f"Unsupported model output type: {type(outputs)}. "
                "Expected ModelOutputs dataclass or (specialist_logits, severity_logits) tuple."
            )

        b_size = input_ids.size(0)
        exporter.add_batch(
            ids=list(batch.get("id", [str(i) for i in range(b_size)])),
            splits=list(batch.get("split", ["unknown"] * b_size)),
            sources=list(batch.get("dataset_source", ["unknown"] * b_size)),
            languages=list(batch.get("language", ["unknown"] * b_size)),
            spec_logits=specialist_logits,
            sev_logits=severity_logits,
            spec_labels=batch["labels_specialist"],
            sev_labels=batch["labels_severity"],
        )

        specialist_true.extend(_tensor_like_list(batch["labels_specialist"]))
        severity_true.extend(_tensor_like_list(batch["labels_severity"]))
        specialist_pred.extend(specialist_logits.argmax(dim=-1).tolist())
        severity_pred.extend(severity_logits.argmax(dim=-1).tolist())

    try:
        exporter.export()
    except Exception as e:
        print(f"Warning: Failed to export predictions: {e}", file=sys.stderr)

    specialist_report = classification_report(
        specialist_true, specialist_pred, num_classes=13
    )
    severity_report = classification_report(
        severity_true,
        severity_pred,
        num_classes=5,
        class_names=[f"S{i}" for i in range(1, 6)],
    )
    severity_confusion = compute_ordinal_confusion(severity_true, severity_pred)

    n_test_rows = len(specialist_true)
    max_rows = getattr(config, "max_rows", None)
    eval_mode = getattr(
        config, "eval_mode", "publication" if max_rows is None else "partial"
    )
    is_full_eval = (max_rows is None) and (eval_mode == "publication")

    # Evaluation Integrity Assertion
    if is_full_eval or eval_mode == "publication":
        if max_rows is not None:
            raise ValueError(
                f"CRITICAL EVALUATION INTEGRITY FAILURE: Publication evaluation must use max_rows=None, but found max_rows={max_rows}."
            )
        if expected_test_rows is not None and n_test_rows != expected_test_rows:
            raise ValueError(
                f"CRITICAL EVALUATION INTEGRITY FAILURE: Publication evaluation expected full test population of {expected_test_rows} rows, but evaluated {n_test_rows} rows."
            )

    return {
        "model_display_name": getattr(
            config, "model_display_name", getattr(model, "display_name", "unknown")
        ),
        "model_short_name": getattr(
            config, "model_short_name", getattr(model, "short_name", "unknown")
        ),
        "is_novel_contribution": bool(getattr(config, "is_novel_contribution", False)),
        "eval_mode": eval_mode,
        "is_full_eval": is_full_eval,
        "max_rows": max_rows,
        "specialist_macro_f1": compute_macro_f1(
            specialist_true, specialist_pred, "specialist"
        ),
        "severity_macro_f1": compute_macro_f1(severity_true, severity_pred, "severity"),
        "specialist_per_class_f1": compute_per_class_f1(
            specialist_true, specialist_pred, "specialist"
        ),
        "severity_per_class_f1": compute_per_class_f1(
            severity_true, severity_pred, [f"S{i}" for i in range(1, 6)]
        ),
        "severity_exact_match_rate": severity_confusion["exact_match_rate"],
        "severity_adjacent_confusion_rate": severity_confusion["adjacent_rate"],
        "severity_distant_confusion_rate": severity_confusion["dangerous_rate"],
        "severity_confusion_matrix": severity_confusion["confusion_matrix"],
        "train_time_seconds": float(getattr(config, "train_time_seconds", 0.0)),
        "evaluated_at": now_utc(),
        "n_test_rows": n_test_rows,
        "expected_test_rows": expected_test_rows,
        "classification_report_specialist": specialist_report,
        "classification_report_severity": severity_report,
    }


def get_system_metadata(config=None, seed=1337) -> dict[str, Any]:
    import platform
    import subprocess

    import torch
    import transformers

    def get_git_info(cmd):
        try:
            return subprocess.check_output(
                cmd, shell=True, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except subprocess.CalledProcessError:
            return "unknown"

    gpu_model = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
    config_dict = config.__dict__.copy() if config else {}
    config_dict = {k: str(v) if k == "model_cls" else v for k, v in config_dict.items()}

    return {
        "git_commit_hash": get_git_info("git rev-parse HEAD"),
        "active_branch": get_git_info("git rev-parse --abbrev-ref HEAD"),
        "timestamp": now_utc(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "None",
        "transformers_version": transformers.__version__,
        "gpu_model": gpu_model,
        "random_seed": seed,
        "experiment_configuration": config_dict,
    }


def _format_report(title: str, report: dict[str, Any]) -> str:
    lines = [title, "=" * len(title)]
    lines.append(f"accuracy: {report['accuracy']:.4f}")
    lines.append(
        f"macro precision/recall/F1: {report['macro_avg']['precision']:.4f} / {report['macro_avg']['recall']:.4f} / {report['macro_avg']['f1']:.4f}"
    )
    lines.append("")
    for row in report["per_class"]:
        lines.append(
            f"{row['class']:<12} precision={row['precision']:.4f} recall={row['recall']:.4f} f1={row['f1']:.4f} support={row['support']}"
        )
    return "\n".join(lines)


def _format_confusion_matrix(matrix: list[list[int]]) -> str:
    if not matrix:
        return ""
    # Determine the width of each column based on the largest number
    max_len = max(len(str(num)) for row in matrix for num in row)
    lines = []
    for row in matrix:
        lines.append(" ".join(str(num).rjust(max_len) for num in row))
    return "\n".join(lines)


def _save_confusion_matrix_png(path: Path, matrix: list[list[int]]) -> None:
    if plt is None:  # pragma: no cover
        return
    values = matrix or [[0 for _ in range(5)] for _ in range(5)]
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(values, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(values)))
    ax.set_yticks(range(len(values)))
    for row_index, row in enumerate(values):
        for col_index, value in enumerate(row):
            ax.text(
                col_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color="black",
                fontsize=8,
            )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_metrics(
    metrics_dict: dict[str, Any], model_short_name: str, config=None, seed=1337
) -> Path:
    result_dir = RESULTS_DIR / model_short_name
    result_dir.mkdir(parents=True, exist_ok=True)

    # Integrity Assertion before saving
    if metrics_dict.get("eval_mode") == "publication" or metrics_dict.get("is_full_eval") is True:
        if metrics_dict.get("max_rows") is not None:
            raise ValueError(
                f"CRITICAL EVALUATION INTEGRITY FAILURE: Cannot save publication metrics with max_rows={metrics_dict.get('max_rows')}."
            )
        if (
            metrics_dict.get("expected_test_rows") is not None
            and metrics_dict.get("n_test_rows") != metrics_dict.get("expected_test_rows")
        ):
            raise ValueError(
                f"CRITICAL EVALUATION INTEGRITY FAILURE: Cannot save publication metrics: n_test_rows ({metrics_dict.get('n_test_rows')}) does not match expected_test_rows ({metrics_dict.get('expected_test_rows')})."
            )

    # Extract config from the dict if provided, else it's passed directly
    if config is None and "experiment_configuration" in metrics_dict:
        config_data = metrics_dict["experiment_configuration"]
    else:
        config_data = config

    meta = get_system_metadata(config=config_data, seed=seed)
    (result_dir / "metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    metrics_path = result_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_dict, indent=2), encoding="utf-8")
    report_text = "\n\n".join(
        [
            _format_report(
                "Specialist Routing",
                metrics_dict.get("classification_report_specialist", {}),
            ),
            _format_report(
                "Severity Triage",
                metrics_dict.get("classification_report_severity", {}),
            ),
        ]
    )
    (result_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")
    # Write confusion matrix as ASCII art
    confusion_matrix = metrics_dict.get("severity_confusion_matrix", [])
    (result_dir / "confusion_matrix.txt").write_text(
        _format_confusion_matrix(confusion_matrix), encoding="utf-8"
    )
    # Also save PNG if matplotlib is available (optional)
    _save_confusion_matrix_png(result_dir / "confusion_matrix.png", confusion_matrix)
    return metrics_path


def load_all_results() -> dict[str, dict]:
    """Load all model results from the results directory.
    Returns a dictionary mapping model short name to metrics dictionary.
    Skips corrupted or missing files with a warning printed to stderr.
    """
    results: dict[str, dict] = {}
    if not RESULTS_DIR.exists():
        return results
    for metrics_path in sorted(RESULTS_DIR.glob("*/metrics.json")):
        model_short_name = metrics_path.parent.name
        try:
            results[model_short_name] = json.loads(
                metrics_path.read_text(encoding="utf-8")
            )
        except (json.JSONDecodeError, OSError) as e:
            print(
                f"Warning: Could not load results for {model_short_name}: {e}",
                file=sys.stderr,
            )
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a MediTriageAI model and save metrics.json."
    )
    return parser


if __name__ == "__main__":
    build_arg_parser().parse_args()
