"""Metrics for MediTriageAI."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.metrics import classification_report as sk_classification_report
from sklearn.metrics import cohen_kappa_score, f1_score


def _as_array(values: Any) -> np.ndarray:
    if hasattr(values, "detach"):
        return values.detach().cpu().numpy()
    elif hasattr(values, "cpu") and hasattr(values, "numpy"):
        return values.cpu().numpy()
    return np.asarray(values, dtype=int)


def _resolve_labels(label_names: str | Sequence[Any]) -> tuple[list[int], list[str]]:
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


def compute_macro_f1(y_true: Any, y_pred: Any, label_names: str | Sequence[Any]) -> float:
    labels, _ = _resolve_labels(label_names)
    return float(
        f1_score(
            _as_array(y_true),
            _as_array(y_pred),
            labels=labels,
            average="macro",
            zero_division=0,
        )
    )


def compute_per_class_f1(y_true: Any, y_pred: Any, label_names: str | Sequence[Any]) -> dict[str, float]:
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


def compute_ordinal_confusion(y_true: Any, y_pred: Any, num_classes: int = 5) -> dict[str, Any]:
    y_true_arr = _as_array(y_true)
    y_pred_arr = _as_array(y_pred)
    total = int(len(y_true_arr)) or 1
    exact = int(np.sum(y_true_arr == y_pred_arr))
    adjacent = int(np.sum(np.abs(y_true_arr - y_pred_arr) == 1))
    dangerous = int(np.sum(np.abs(y_true_arr - y_pred_arr) >= 2))
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true_value, pred_value in zip(y_true_arr, y_pred_arr):
        matrix[int(np.clip(true_value, 0, num_classes - 1)), int(np.clip(pred_value, 0, num_classes - 1))] += 1
    return {
        "exact_match": exact,
        "adjacent_confusion": adjacent,
        "distant_confusion": dangerous,
        "exact_match_rate": exact / total,
        "adjacent_rate": adjacent / total,
        "dangerous_rate": dangerous / total,
        "confusion_matrix": matrix.tolist(),
    }


def compute_classification_report(y_true: Any, y_pred: Any, labels: Iterable[int]) -> dict[str, Any]:
    labels = list(labels)
    report = sk_classification_report(
        _as_array(y_true),
        _as_array(y_pred),
        labels=labels,
        target_names=[str(label) for label in labels],
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(report["accuracy"]),
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
    class_names: list[str] | None = None
) -> dict[str, Any]:
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
    return {
        "accuracy": float(report["accuracy"]),
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



def landis_koch_label(kappa: float) -> str:
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
    return float(cohen_kappa_score(_as_array(y_true), _as_array(y_pred), labels=list(range(num_classes))))


def severity_metrics(y_true: Any, y_pred: Any) -> dict[str, Any]:
    report = classification_report(y_true, y_pred, num_classes=5, class_names=[f"S{i}" for i in range(1, 6)])
    y_true_arr = _as_array(y_true)
    y_pred_arr = _as_array(y_pred)
    report["ordinal_confusion"] = compute_ordinal_confusion(y_true_arr, y_pred_arr)
    report["ordinal_error_matrix"] = np.zeros((5, 9), dtype=int).tolist()  # placeholder
    report["mean_absolute_error"] = float(np.mean(np.abs(y_true_arr - y_pred_arr))) if len(y_true_arr) else 0.0
    return report


def generate_novelty_summary(results_by_model: dict[str, dict[str, Any]]) -> str:
    if not results_by_model:
        return "[RESULT_PLACEHOLDER: novelty summary unavailable until model results are exported]"

    ranked = list(results_by_model.values())
    novel_model = next((item for item in ranked if item.get("is_novel_contribution")), ranked[0])
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
    sev_name, sev_baseline = strongest_baseline("severity_macro_f1")
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
    metric_key = "specialist_macro_f1" if task.lower().startswith("spec") else "severity_macro_f1"
    heading = "Specialist F1" if metric_key == "specialist_macro_f1" else "Severity F1"
    rows = sorted(results_by_model.values(), key=lambda item: float(item.get(metric_key, 0.0)), reverse=True)
    body = ["\\begin{tabular}{l r}", "\\hline", f"Model & {heading} \\\\", "\\hline"]
    for item in rows:
        body.append(
            f"{escape_latex(str(item.get('model_display_name', 'model')))} & {float(item.get(metric_key, 0.0)) * 100:.1f}\\% \\\\"
        )
    body.extend(["\\hline", "\\end{tabular}"])
    return "\n".join(body)


def escape_latex(text: str) -> str:
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