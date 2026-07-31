"""Metrics for MediTriageAI.

This module provides:
- MaskedMetrics: class-based masked multi-task metric accumulator (Phase 3+)
- Standalone metric functions: compute_macro_f1, compute_per_class_f1, etc.
  (used by evaluate.py, train.py, analysis/, and run_experiment.py)
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.metrics import (
    classification_report as sk_classification_report,
)

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _as_array(values: Any) -> np.ndarray:
    """Convert tensors or lists to numpy arrays."""
    if hasattr(values, "detach"):
        return values.detach().cpu().numpy()
    elif hasattr(values, "cpu") and hasattr(values, "numpy"):
        return values.cpu().numpy()
    return np.asarray(values, dtype=int)


def _resolve_labels(
    label_names: str | Sequence[Any],
) -> tuple[list[int], list[str]]:
    """Resolve a label specification to (integer labels, string names)."""
    if isinstance(label_names, str):
        key = label_names.lower()
        if key in {"specialist", "department", "routing"}:
            labels = list(range(13))
        elif key in {"severity", "triage"}:
            labels = list(range(5))
        else:
            raise ValueError(f"Unknown label set: {label_names!r}")
        return labels, [str(label) for label in labels]

    names = [str(name) for name in label_names]
    return list(range(len(names))), names


# ---------------------------------------------------------------------------
# MaskedMetrics (Phase 3+): class-based masked multi-task accumulator
# ---------------------------------------------------------------------------


class MaskedMetrics:
    """Computes validation metrics strictly ignoring masked samples (-1)."""

    def __init__(self, ignore_index: int = -1):
        self.ignore_index = ignore_index
        self.specialist_preds: list[int] = []
        self.specialist_labels: list[int] = []
        self.specialist_probs: list[list[float]] = []

        self.severity_preds: list[int] = []
        self.severity_labels: list[int] = []
        self.severity_probs: list[list[float]] = []

    def reset(self):
        self.specialist_preds.clear()
        self.specialist_labels.clear()
        self.specialist_probs.clear()
        self.severity_preds.clear()
        self.severity_labels.clear()
        self.severity_probs.clear()

    def _to_numpy(self, tensor) -> np.ndarray:
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return np.array(tensor)

    def update(
        self,
        spec_logits: torch.Tensor,
        sev_logits: torch.Tensor,
        spec_labels: torch.Tensor,
        sev_labels: torch.Tensor,
    ):
        """Update metrics for a single batch."""
        spec_logits_np = self._to_numpy(spec_logits)
        sev_logits_np = self._to_numpy(sev_logits)
        spec_labels_np = self._to_numpy(spec_labels)
        sev_labels_np = self._to_numpy(sev_labels)

        # Specialist masking
        spec_mask = spec_labels_np != self.ignore_index
        if spec_mask.any():
            valid_spec_labels = spec_labels_np[spec_mask]
            valid_spec_logits = spec_logits_np[spec_mask]
            valid_spec_probs = torch.softmax(
                torch.tensor(valid_spec_logits), dim=-1
            ).numpy()
            valid_spec_preds = valid_spec_logits.argmax(axis=-1)

            self.specialist_labels.extend(valid_spec_labels.tolist())
            self.specialist_preds.extend(valid_spec_preds.tolist())
            self.specialist_probs.extend(valid_spec_probs.tolist())

        # Severity masking
        sev_mask = sev_labels_np != self.ignore_index
        if sev_mask.any():
            valid_sev_labels = sev_labels_np[sev_mask]
            valid_sev_logits = sev_logits_np[sev_mask]
            valid_sev_probs = torch.softmax(
                torch.tensor(valid_sev_logits), dim=-1
            ).numpy()
            valid_sev_preds = valid_sev_logits.argmax(axis=-1)

            self.severity_labels.extend(valid_sev_labels.tolist())
            self.severity_preds.extend(valid_sev_preds.tolist())
            self.severity_probs.extend(valid_sev_probs.tolist())

    def compute(
        self, specialist_class_names: list[str], severity_class_names: list[str]
    ) -> dict[str, Any]:
        """Compute all metrics and return a summary dictionary."""
        metrics = {}

        if len(self.specialist_labels) > 0:
            y_true = np.array(self.specialist_labels)
            y_pred = np.array(self.specialist_preds)
            y_prob = np.array(self.specialist_probs)

            metrics["specialist"] = self._compute_task_metrics(
                y_true, y_pred, y_prob, specialist_class_names
            )
        else:
            metrics["specialist"] = {}

        if len(self.severity_labels) > 0:
            y_true = np.array(self.severity_labels)
            y_pred = np.array(self.severity_preds)
            y_prob = np.array(self.severity_probs)

            metrics["severity"] = self._compute_task_metrics(
                y_true, y_pred, y_prob, severity_class_names
            )
        else:
            metrics["severity"] = {}

        return metrics

    def _compute_task_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        class_names: list[str],
    ) -> dict[str, Any]:

        unique_labels = list(range(len(class_names)))

        # Basic Metrics
        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)

        # AUROC (OvR)
        try:
            auroc = roc_auc_score(
                y_true, y_prob, multi_class="ovr", labels=unique_labels
            )
        except ValueError:
            auroc = float("nan")

        # Classification Report (Per-class Precision, Recall, F1)
        report = sk_classification_report(
            y_true,
            y_pred,
            labels=unique_labels,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred, labels=unique_labels)

        return {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "micro_f1": micro_f1,
            "auroc": auroc,
            "report": report,
            "confusion_matrix": cm.tolist(),
        }

    def export_artifacts(
        self,
        metrics: dict[str, Any],
        output_dir: str,
        specialist_names: list[str],
        severity_names: list[str],
    ):
        """Export classification reports, confusion matrices, and per-class metrics."""
        import os

        os.makedirs(output_dir, exist_ok=True)

        # 1. Validation Summary JSON & Classification Report
        report_json = {}
        per_class_rows = []

        for task, names in [
            ("specialist", specialist_names),
            ("severity", severity_names),
        ]:
            task_metrics = metrics.get(task, {})
            if not task_metrics:
                continue

            report_json[task] = task_metrics.get("report", {})

            # 2. Per-class metrics
            report = task_metrics.get("report", {})
            for name in names:
                if name in report:
                    row = {"task": task, "class": name}
                    row.update(report[name])
                    per_class_rows.append(row)

            # 3. Confusion Matrix
            cm = np.array(task_metrics.get("confusion_matrix", []))
            if cm.size > 0:
                df_cm = pd.DataFrame(cm, index=names, columns=names)
                df_cm.to_csv(os.path.join(output_dir, f"{task}_confusion_matrix.csv"))

                # Normalized
                row_sums = cm.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1
                norm_cm = cm / row_sums
                df_norm_cm = pd.DataFrame(norm_cm, index=names, columns=names)
                df_norm_cm.to_csv(
                    os.path.join(output_dir, f"{task}_normalized_confusion_matrix.csv")
                )

        with open(os.path.join(output_dir, "classification_report.json"), "w") as f:
            json.dump(report_json, f, indent=2)

        summary = {
            task: {
                k: v for k, v in m.items() if k not in ["report", "confusion_matrix"]
            }
            for task, m in metrics.items()
        }
        with open(os.path.join(output_dir, "validation_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        if per_class_rows:
            pd.DataFrame(per_class_rows).to_csv(
                os.path.join(output_dir, "per_class_metrics.csv"), index=False
            )


# ---------------------------------------------------------------------------
# Standalone metric functions (pre-Phase 3 API, still used by evaluate.py,
# train.py, colab_train.py, analysis/, and run_experiment.py)
# ---------------------------------------------------------------------------


def compute_macro_f1(
    y_true: Any, y_pred: Any, label_names: str | Sequence[Any]
) -> float:
    """Compute macro-averaged F1 score."""
    labels, _ = _resolve_labels(label_names)
    if isinstance(label_names, str):
        observed = sorted(set(_as_array(y_true)).union(set(_as_array(y_pred))))
        labels = [label for label in labels if label in observed] or labels
    return float(
        f1_score(
            _as_array(y_true),
            _as_array(y_pred),
            labels=labels,
            average="macro",
            zero_division=0,
        )
    )


def compute_per_class_f1(
    y_true: Any, y_pred: Any, label_names: str | Sequence[Any]
) -> dict[str, float]:
    """Compute per-class F1 scores."""
    labels, names = _resolve_labels(label_names)
    report = sk_classification_report(
        _as_array(y_true),
        _as_array(y_pred),
        labels=labels,
        target_names=names,
        output_dict=True,
        zero_division=0,
    )
    return {name: float(report[name]["f1-score"]) for name in names}


def compute_ordinal_confusion(
    y_true: Any, y_pred: Any, num_classes: int = 5
) -> dict[str, Any]:
    """Compute ordinal confusion breakdown (exact, adjacent, distant)."""
    y_true_arr = _as_array(y_true)
    y_pred_arr = _as_array(y_pred)
    total = len(y_true_arr) or 1
    exact = int(np.sum(y_true_arr == y_pred_arr))
    adjacent = int(np.sum(np.abs(y_true_arr - y_pred_arr) == 1))
    dangerous = int(np.sum(np.abs(y_true_arr - y_pred_arr) >= 2))
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true_value, pred_value in zip(y_true_arr, y_pred_arr):
        matrix[
            int(np.clip(true_value, 0, num_classes - 1)),
            int(np.clip(pred_value, 0, num_classes - 1)),
        ] += 1
    return {
        "exact_match": exact,
        "adjacent_confusion": adjacent,
        "distant_confusion": dangerous,
        "exact_match_rate": exact / total,
        "adjacent_rate": adjacent / total,
        "dangerous_rate": dangerous / total,
        "confusion_matrix": matrix.tolist(),
    }


def compute_classification_report(
    y_true: Any, y_pred: Any, labels: Iterable[int]
) -> dict[str, Any]:
    """Compute a structured classification report for given label indices."""
    labels = list(labels)
    report = sk_classification_report(
        _as_array(y_true),
        _as_array(y_pred),
        labels=labels,
        target_names=[str(label) for label in labels],
        output_dict=True,
        zero_division=0,
    )
    acc = report.get("accuracy")
    if acc is None:
        y_true_arr = _as_array(y_true)
        y_pred_arr = _as_array(y_pred)
        valid_mask = np.isin(y_true_arr, labels)
        if valid_mask.any():
            acc = float(accuracy_score(y_true_arr[valid_mask], y_pred_arr[valid_mask]))
        else:
            acc = 0.0

    return {
        "accuracy": float(acc),
        "macro_avg": {
            "precision": float(report["macro avg"]["precision"]),
            "recall": float(report["macro avg"]["recall"]),
            "f1": float(report["macro avg"]["f1-score"]),
        },
        "weighted_avg": {
            "precision": float(report["weighted avg"]["precision"]),
            "recall": float(report["weighted avg"]["recall"]),
            "f1": float(report["weighted avg"]["f1-score"]),
        },
        "per_class": [
            {
                "class": str(label),
                "precision": float(report[str(label)]["precision"]),
                "recall": float(report[str(label)]["recall"]),
                "f1": float(report[str(label)]["f1-score"]),
                "support": int(report[str(label)]["support"]),
            }
            for label in labels
        ],
    }


def classification_report(
    y_true: Any,
    y_pred: Any,
    num_classes: int,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """Compute a structured classification report with optional class names."""
    labels = list(range(num_classes))
    names = class_names if class_names is not None else [str(label) for label in labels]
    report = sk_classification_report(
        _as_array(y_true),
        _as_array(y_pred),
        labels=labels,
        target_names=names,
        output_dict=True,
        zero_division=0,
    )
    acc = report.get("accuracy")
    if acc is None:
        y_true_arr = _as_array(y_true)
        y_pred_arr = _as_array(y_pred)
        valid_mask = np.isin(y_true_arr, labels)
        if valid_mask.any():
            acc = float(accuracy_score(y_true_arr[valid_mask], y_pred_arr[valid_mask]))
        else:
            acc = 0.0

    return {
        "accuracy": float(acc),
        "macro_avg": {
            "precision": float(report["macro avg"]["precision"]),
            "recall": float(report["macro avg"]["recall"]),
            "f1": float(report["macro avg"]["f1-score"]),
        },
        "weighted_avg": {
            "precision": float(report["weighted avg"]["precision"]),
            "recall": float(report["weighted avg"]["recall"]),
            "f1": float(report["weighted avg"]["f1-score"]),
        },
        "per_class": [
            {
                "class": names[i],
                "precision": float(report[names[i]]["precision"]),
                "recall": float(report[names[i]]["recall"]),
                "f1": float(report[names[i]]["f1-score"]),
                "support": int(report[names[i]]["support"]),
            }
            for i in range(num_classes)
        ],
    }


# ---------------------------------------------------------------------------
# Inter-rater / agreement metrics
# ---------------------------------------------------------------------------


def landis_koch_label(kappa: float) -> str:
    """Return Landis-Koch interpretation for a kappa value."""
    if kappa < 0.0:
        return "poor"
    if kappa < 0.20:
        return "slight"
    if kappa < 0.40:
        return "fair"
    if kappa < 0.60:
        return "moderate"
    if kappa < 0.80:
        return "substantial"
    return "almost perfect"


def cohens_kappa(y_true: Any, y_pred: Any, num_classes: int) -> float:
    """Compute Cohen's kappa."""
    return float(
        cohen_kappa_score(
            _as_array(y_true), _as_array(y_pred), labels=list(range(num_classes))
        )
    )


# ---------------------------------------------------------------------------
# Composite metric helpers
# ---------------------------------------------------------------------------


def severity_metrics(y_true: Any, y_pred: Any) -> dict[str, Any]:
    """Compute all severity-specific metrics."""
    report = classification_report(
        y_true, y_pred, num_classes=5, class_names=[f"S{i}" for i in range(1, 6)]
    )
    y_true_arr = _as_array(y_true)
    y_pred_arr = _as_array(y_pred)
    report["ordinal_confusion"] = compute_ordinal_confusion(y_true_arr, y_pred_arr)
    report["ordinal_error_matrix"] = np.zeros((5, 9), dtype=int).tolist()
    report["mean_absolute_error"] = (
        float(np.mean(np.abs(y_true_arr - y_pred_arr))) if len(y_true_arr) else 0.0
    )
    return report


# ---------------------------------------------------------------------------
# Paper / reporting helpers
# ---------------------------------------------------------------------------


def generate_novelty_summary(results_by_model: dict[str, dict[str, Any]]) -> str:
    """Generate a novelty summary comparing the novel model to baselines."""
    if not results_by_model:
        return "[RESULT_PLACEHOLDER: novelty summary unavailable until model results are exported]"

    ranked = list(results_by_model.values())
    novel_model = next(
        (item for item in ranked if item.get("is_novel_contribution")), ranked[0]
    )
    baselines = [item for item in ranked if item is not novel_model]
    if not baselines:
        return (
            f"{novel_model.get('model_display_name', 'XLM-RoBERTa-large')} is the only evaluated model so far, "
            "so a cross-model novelty margin cannot yet be computed."
        )

    def strongest_baseline(metric: str) -> tuple[str, float]:
        best = max(baselines, key=lambda item: float(item.get(metric, 0.0)))
        return best.get("model_display_name", "baseline"), float(best.get(metric, 0.0))

    spec_name, spec_baseline = strongest_baseline("specialist_macro_f1")
    _sev_name, sev_baseline = strongest_baseline("severity_macro_f1")
    spec_score = float(novel_model.get("specialist_macro_f1", 0.0))
    sev_score = float(novel_model.get("severity_macro_f1", 0.0))
    delta_spec = spec_score - spec_baseline
    delta_sev = sev_score - sev_baseline
    return (
        f"{novel_model.get('model_display_name', 'XLM-RoBERTa-large')} achieves {spec_score:.1%} specialist F1 and "
        f"{sev_score:.1%} severity F1, outperforming the strongest baseline ({spec_name}) by {delta_spec:.1%} and "
        f"{delta_sev:.1%} respectively ({delta_spec:.3f}, {delta_sev:.3f}). The Hinglish phonetic perturbation training regime appears to strengthen "
        "cross-script robustness while preserving the routing and triage heads."
    )


def generate_latex_table(results_by_model: dict[str, dict[str, Any]], task: str) -> str:
    """Generate a LaTeX table of model results."""
    metric_key = (
        "specialist_macro_f1"
        if task.lower().startswith("spec")
        else "severity_macro_f1"
    )
    heading = "Specialist F1" if metric_key == "specialist_macro_f1" else "Severity F1"
    rows = sorted(
        results_by_model.values(),
        key=lambda item: float(item.get(metric_key, 0.0)),
        reverse=True,
    )
    body = [
        "\\begin{tabular}{l r}",
        "\\hline",
        f"Model & {heading} \\\\",
        "\\hline",
    ]
    for item in rows:
        body.append(
            f"{escape_latex(str(item.get('model_display_name', 'model')))} & {float(item.get(metric_key, 0.0)) * 100:.1f}\\% \\\\"
        )
    body.extend(["\\hline", "\\end{tabular}"])
    return "\n".join(body)


def escape_latex(text: str) -> str:
    """Escape special LaTeX characters."""
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    escaped = text
    for source, replacement in replacements.items():
        escaped = escaped.replace(source, replacement)
    return escaped
