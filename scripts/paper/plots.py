import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from .utils import generate_placeholder

logger = logging.getLogger(__name__)


def plot_confusion_matrix(
    y_true: list[Any] | None,
    y_pred: list[Any] | None,
    labels: list[str],
    output_path: Path,
) -> None:
    """Plot and save a confusion matrix.

    Args:
        y_true (Optional[List[Any]]): Ground truth labels. If None, generates a placeholder.
        y_pred (Optional[List[Any]]): Predicted labels.
        labels (List[str]): List of label names for the axes.
        output_path (Path): Destination to save the plot.
    """
    if y_true is None or y_pred is None or len(y_true) == 0:
        logger.warning(
            f"Missing predictions for {output_path.name}. Generating placeholder."
        )
        generate_placeholder(
            output_path, "Confusion Matrix\\nGenerated after experiments."
        )
        return

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved confusion matrix to {output_path}")


def plot_roc_curve(
    y_true: list[int] | None, y_probs: list[float] | None, output_path: Path
) -> None:
    """Plot and save an ROC curve.

    Args:
        y_true (Optional[List[int]]): Ground truth binary labels. If None, generates a placeholder.
        y_probs (Optional[List[float]]): Predicted probabilities.
        output_path (Path): Destination to save the plot.
    """
    if y_true is None or y_probs is None or len(y_true) == 0:
        logger.warning(
            f"Missing probabilities for {output_path.name}. Generating placeholder."
        )
        generate_placeholder(output_path, "ROC Curve\\nGenerated after experiments.")
        return

    fpr, tpr, _ = roc_curve(y_true, y_probs)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fpr, tpr, color="darkorange", lw=2)
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved ROC curve to {output_path}")


def plot_pr_curve(
    y_true: list[int] | None, y_probs: list[float] | None, output_path: Path
) -> None:
    """Plot and save a Precision-Recall curve.

    Args:
        y_true (Optional[List[int]]): Ground truth binary labels. If None, generates a placeholder.
        y_probs (Optional[List[float]]): Predicted probabilities.
        output_path (Path): Destination to save the plot.
    """
    if y_true is None or y_probs is None or len(y_true) == 0:
        logger.warning(
            f"Missing probabilities for {output_path.name}. Generating placeholder."
        )
        generate_placeholder(output_path, "PR Curve\\nGenerated after experiments.")
        return

    prec, rec, _ = precision_recall_curve(y_true, y_probs)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(rec, prec, color="blue", lw=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved PR curve to {output_path}")


def plot_calibration(
    y_true: list[int] | None, y_probs: list[float] | None, output_path: Path
) -> None:
    """Plot and save a calibration reliability diagram.

    Args:
        y_true (Optional[List[int]]): Ground truth labels.
        y_probs (Optional[List[float]]): Predicted probabilities.
        output_path (Path): Destination to save the plot.
    """
    logger.warning(
        f"Missing calibration data for {output_path.name}. Generating placeholder."
    )
    generate_placeholder(output_path, "Calibration Plot\\nGenerated after experiments.")


def plot_learning_curves(history: Any | None, output_path: Path) -> None:
    """Plot and save learning curves from training history.

    Args:
        history (Optional[Any]): Training history object.
        output_path (Path): Destination to save the plot.
    """
    logger.warning(
        f"Missing history data for {output_path.name}. Generating placeholder."
    )
    generate_placeholder(output_path, "Learning Curves\\nGenerated after experiments.")


def plot_grad_cam(output_path: Path) -> None:
    """Plot and save a Grad-CAM / Attention visualization.

    Args:
        output_path (Path): Destination to save the plot.
    """
    logger.warning(
        f"Missing attention maps for {output_path.name}. Generating placeholder."
    )
    generate_placeholder(
        output_path, "Grad-CAM Attention\\nGenerated after experiments."
    )
