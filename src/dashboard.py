"""Dashboard helpers for MediTriageAI training output."""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn
from rich.table import Table


def make_epoch_progress() -> Progress:
    return Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.0f}%",
        "•",
        TaskProgressColumn(),
        "•",
        TimeRemainingColumn(),
    )


def make_val_progress() -> Progress:
    return Progress(TextColumn("[bold green]Validating..."), BarColumn(bar_width=None), "[progress.percentage]{task.percentage:>3.0f}%")


def build_metrics_table(metrics: dict[str, float], epoch: int, lr: float) -> Table:
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="dim", width=20)
    table.add_column("Value", justify="right")
    table.add_row("Epoch", f"{epoch + 1}")
    table.add_row("Loss", f"{metrics['loss']:.4f}")
    table.add_row("Specialist Loss", f"{metrics['specialist_loss']:.4f}")
    table.add_row("Severity Loss", f"{metrics['severity_loss']:.4f}")
    table.add_row("Specialist Acc", f"{metrics['specialist_acc']:.2%}")
    table.add_row("Severity Acc", f"{metrics['severity_acc']:.2%}")
    table.add_row("Learning Rate", f"{lr:.2e}")
    return table


def build_val_summary_table(epoch: int, val_metrics: dict[str, float], elapsed_s: float) -> Table:
    table = Table(show_header=True, header_style="bold green")
    table.add_column("Metric", style="dim", width=25)
    table.add_column("Value", justify="right")
    table.add_row("Epoch", f"{epoch + 1} (Validation)")
    table.add_row("Val Loss", f"{val_metrics['loss']:.4f}")
    table.add_row("Val Specialist Acc", f"{val_metrics['specialist_acc']:.2%}")
    table.add_row("Val Severity Acc", f"{val_metrics['severity_acc']:.2%}")
    table.add_row("Time Elapsed", f"{elapsed_s:.1f}s")
    return table
