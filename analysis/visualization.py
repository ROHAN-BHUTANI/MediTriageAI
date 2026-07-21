"""Plotting and visualization utilities for the MediTriageAI analysis framework."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_confusion_matrix(
    cm: np.ndarray,
    classes: list[str],
    title: str,
    save_path: Path,
    dpi: int = 300
) -> None:
    """Plot and save a high-resolution confusion matrix heatmap.
    
    Args:
        cm: Confusion matrix array.
        classes: List of class names for labels.
        title: Title of the plot.
        save_path: Path to save the PNG file.
        dpi: Dots per inch for image resolution.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7), dpi=dpi)
    
    # Choose a nice professional blue colormap
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        ax=ax,
        cbar=True,
        square=True
    )
    
    ax.set_title(title, fontsize=12, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted Label", fontsize=10, labelpad=10)
    ax.set_ylabel("True Label", fontsize=10, labelpad=10)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)
    plt.close(fig)


def plot_reliability_diagram(
    reliability_data: dict[str, list[float]],
    ece: float,
    title: str,
    save_path: Path,
    dpi: int = 300
) -> None:
    """Plot and save a reliability diagram for calibration analysis.
    
    Args:
        reliability_data: Dictionary returned by get_reliability_curve_data.
        ece: Expected Calibration Error value.
        title: Title of the plot.
        save_path: Path to save the PNG file.
        dpi: Resolution.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=dpi)
    
    conf = reliability_data["bin_confidences"]
    acc = reliability_data["bin_accuracies"]
    
    # Plot diagonal reference representing perfect calibration
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    
    # Plot model calibration curve
    ax.plot(conf, acc, marker="o", color="#1f77b4", linewidth=2, label=f"Model (ECE={ece:.4f})")
    
    # Plot identity gap shading
    ax.fill_between(conf, conf, acc, color="red", alpha=0.1, label="Calibration Gap")
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence", fontsize=10)
    ax.set_ylabel("Accuracy", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)
    plt.close(fig)


def plot_confidence_histogram(
    confidences: np.ndarray,
    mean_conf: float,
    accuracy: float,
    title: str,
    save_path: Path,
    dpi: int = 300
) -> None:
    """Plot and save a confidence distribution histogram.
    
    Args:
        confidences: 1D array of predicted confidences.
        mean_conf: Mean confidence value.
        accuracy: Overall accuracy value.
        title: Title of the plot.
        save_path: Path to save the PNG file.
        dpi: Resolution.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=dpi)
    
    # Plot histogram
    sns.histplot(confidences, bins=15, kde=False, color="#a6c8e0", edgecolor="white", ax=ax, stat="probability")
    
    # Mark average confidence and accuracy
    ax.axvline(mean_conf, color="red", linestyle="-", linewidth=1.5, label=f"Mean Conf ({mean_conf:.3f})")
    ax.axvline(accuracy, color="green", linestyle="--", linewidth=1.5, label=f"Accuracy ({accuracy:.3f})")
    
    ax.set_xlim(0, 1)
    ax.set_xlabel("Confidence", fontsize=10)
    ax.set_ylabel("Probability", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)
    plt.close(fig)


def plot_agreement_heatmap(
    df_matrix: pd.DataFrame,
    title: str,
    save_path: Path,
    dpi: int = 300
) -> None:
    """Plot and save a cross-model agreement heatmap.
    
    Args:
        df_matrix: Pairwise matrix DataFrame (e.g. Cohen's Kappa).
        title: Title of the plot.
        save_path: Path to save.
        dpi: Resolution.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6), dpi=dpi)
    
    sns.heatmap(
        df_matrix,
        annot=True,
        fmt=".3f",
        cmap="GnBu",
        ax=ax,
        cbar=True,
        square=True,
        vmin=0.0,
        vmax=1.0
    )
    
    ax.set_title(title, fontsize=12, fontweight="bold", pad=15)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)
    plt.close(fig)


def plot_dataset_distributions(
    df_test: pd.DataFrame,
    figures_dir: Path,
    dpi: int = 300
) -> dict[str, Path]:
    """Generate publication-ready figures representing the dataset distribution.
    
    Generates:
        1. Class frequency histogram
        2. Rare-class distribution
        3. Long-tail label distribution
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    
    # 1. Class frequency histogram (Specialist)
    spec_counts = df_test["department_code"].value_counts()
    
    fig, ax = plt.subplots(figsize=(9, 5), dpi=dpi)
    sns.barplot(x=spec_counts.index, y=spec_counts.values, hue=spec_counts.index, legend=False, palette="viridis", ax=ax)
    ax.set_title("Dataset Specialist Class Frequencies (Test Split)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Specialist Department", fontsize=10)
    ax.set_ylabel("Sample Count", fontsize=10)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    freq_path = figures_dir / "class_frequency.png"
    plt.savefig(freq_path, dpi=dpi)
    plt.close(fig)
    paths["class_frequency"] = freq_path
    
    # 2. Long-tail distribution
    fig, ax = plt.subplots(figsize=(9, 5), dpi=dpi)
    # Sort in descending order
    sorted_counts = spec_counts.sort_values(ascending=False)
    ax.plot(range(len(sorted_counts)), sorted_counts.values, marker="o", color="#d62728", linewidth=2)
    ax.fill_between(range(len(sorted_counts)), sorted_counts.values, color="#d62728", alpha=0.1)
    ax.set_title("Long-tail Specialist Label Distribution", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Specialist Rank", fontsize=10)
    ax.set_ylabel("Sample Count", fontsize=10)
    ax.set_xticks(range(len(sorted_counts)))
    ax.set_xticklabels(sorted_counts.index, rotation=45, ha="right", fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    lt_path = figures_dir / "long_tail_distribution.png"
    plt.savefig(lt_path, dpi=dpi)
    plt.close(fig)
    paths["long_tail"] = lt_path

    # 3. Rare-class distribution (frequencies below the lower quartile of frequencies)
    q25 = np.percentile(spec_counts.values, 25)
    rare_mask = spec_counts <= q25
    rare_counts = spec_counts[rare_mask]
    
    fig, ax = plt.subplots(figsize=(7, 4), dpi=dpi)
    if not rare_counts.empty:
        sns.barplot(x=rare_counts.index, y=rare_counts.values, hue=rare_counts.index, legend=False, palette="magma", ax=ax)
    ax.set_title(f"Rare-Class Specialist Distribution (Support <= {int(q25)})", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Rare Specialist Class", fontsize=10)
    ax.set_ylabel("Sample Count", fontsize=10)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    rare_path = figures_dir / "rare_class_distribution.png"
    plt.savefig(rare_path, dpi=dpi)
    plt.close(fig)
    paths["rare_class"] = rare_path
    
    return paths
