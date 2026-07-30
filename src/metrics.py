import json
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


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
        # Handle cases where not all classes are present in y_true
        try:
            auroc = roc_auc_score(
                y_true, y_prob, multi_class="ovr", labels=unique_labels
            )
        except ValueError:
            auroc = float("nan")  # E.g., if a class is entirely missing from y_true

        # Classification Report (Per-class Precision, Recall, F1)
        report = classification_report(
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
                # avoid division by zero
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
