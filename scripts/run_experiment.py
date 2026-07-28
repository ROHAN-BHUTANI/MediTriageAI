"""Interactive experiment runner for MediTriageAI."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from dataclasses import dataclass

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.distilbert_multi import DistilBertMultilingualModel
from models.indic_bert import IndicBertModel
from models.mbert import MBertModel
from models.xlm_roberta import XLMRobertaLargeModel
from models.emergent_path_triage import EmergentPathTriageModel
from scripts import evaluate as evaluator
from scripts import export_dashboard_data as dashboard_exporter
from scripts import train as trainer
from src.metrics import generate_novelty_summary

RESULTS_DIR = REPO_ROOT / "results"


@dataclass(frozen=True)
class ExperimentModel:
    choice: int
    model_cls: type


MODEL_ZOO = (
    ExperimentModel(1, XLMRobertaLargeModel),
    ExperimentModel(2, MBertModel),
    ExperimentModel(3, DistilBertMultilingualModel),
    ExperimentModel(4, IndicBertModel),
    ExperimentModel(5, EmergentPathTriageModel),
)



def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def discover_test_count() -> int:
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "tests", "--collect-only", "-q"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        return sum(1 for line in completed.stdout.splitlines() if line.strip() and "::" in line)
    except Exception:
        return sum(1 for _ in REPO_ROOT.glob("tests/test_*.py"))


def header_panel() -> Panel:
    body = (
        f"MediTriageAI — Experiment Runner\n"
        f"Date: {now_utc().split('T')[0]}   "
        f"Tests passing: {discover_test_count()}   "
        f"Dataset: 19,996 rows\n\n"
        "[1] XLM-RoBERTa-large        Baseline\n"
        "[2] mBERT                    Multilingual baseline\n"
        "[3] DistilBERT-multilingual  Lightweight ablation\n"
        "[4] IndicBERT                Hindi-specialist baseline\n"
        "[5] E-PATH-CO-REASON         * Novel contribution\n"
        "[6] All models (sequential)\n"
        "[7] Show comparison table (no training)\n"
        "[8] Export dashboard data"
    )
    return Panel(body, border_style="blue")


def load_metrics_files(results_dir: Path = RESULTS_DIR) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    if not results_dir.exists():
        return results
    for metrics_path in sorted(results_dir.glob("*/metrics.json")):
        try:
            results[metrics_path.parent.name] = json.loads(metrics_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue

    # Prevent mixing historical or differently-filtered evaluations
    if results:
        latest_eval = max(results.values(), key=lambda x: x.get("evaluated_at", ""))
        expected_rows = latest_eval.get("n_test_rows")
        results = {k: v for k, v in results.items() if v.get("n_test_rows") == expected_rows}

    return results


def _metric_value(item: dict[str, Any], *keys: str) -> float:
    for key in keys:
        value = item.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


def _best_slug(results: dict[str, dict[str, Any]], metric_keys: tuple[str, ...]) -> str:
    if not results:
        return ""
    return max(results.items(), key=lambda pair: _metric_value(pair[1], *metric_keys))[0]


def build_comparison_table(results: dict[str, dict[str, Any]]) -> Table:
    table = Table(title="MediTriageAI — Model Comparison Report", show_lines=False)
    table.add_column("Model")
    table.add_column("Spec F1", justify="right")
    table.add_column("Sev F1", justify="right")
    table.add_column("Adj Err", justify="right")
    table.add_column("Role")

    if not results:
        table.add_row("[NOT RUN]", "[NOT RUN]", "[NOT RUN]", "[NOT RUN]")
        return table

    best_spec = _best_slug(results, ("specialist_macro_f1", "specialist_f1"))
    best_sev = _best_slug(results, ("severity_macro_f1", "severity_f1"))
    best_adj_err = min(results.items(), key=lambda pair: _metric_value(pair[1], "severity_adjacent_confusion_rate", "severity_adjacent_confusion"))[0]

    ordered = sorted(results.items(), key=lambda pair: _metric_value(pair[1], "specialist_macro_f1", "specialist_f1"), reverse=True)
    for slug, item in ordered:
        spec_text = f"{_metric_value(item, 'specialist_macro_f1', 'specialist_f1'):.3f}"
        sev_text = f"{_metric_value(item, 'severity_macro_f1', 'severity_f1'):.3f}"
        adj_err_text = f"{_metric_value(item, 'severity_adjacent_confusion_rate', 'severity_adjacent_confusion'):.3f}"

        if slug == best_spec:
            spec_text = f"[green]{spec_text}[/green]"
        if slug == best_sev:
            sev_text = f"[green]{sev_text}[/green]"
        if slug == best_adj_err:
            adj_err_text = f"[green]{adj_err_text}[/green]"

        role = "* Novel" if item.get("is_novel_contribution") else "Baseline"
        table.add_row(
            item.get("model_display_name", slug),
            spec_text,
            sev_text,
            adj_err_text,
            role
        )
    return table


def build_novelty_paragraph(results: dict[str, dict[str, Any]]) -> str:
    return generate_novelty_summary(results)


def show_comparison_report(console: Console, results: dict[str, dict[str, Any]]) -> None:
    console.print(build_comparison_table(results))
    if results:
        console.print(f"Novelty summary: {build_novelty_paragraph(results)}")
    else:
        console.print("[dim]Novelty summary: [RESULT_PLACEHOLDER: novelty summary unavailable until model results are exported][/dim]")


def model_for_choice(choice: int) -> ExperimentModel | None:
    return next((spec for spec in MODEL_ZOO if spec.choice == choice), None)


def _model_summary_line(model_cls: type) -> str:
    notes = model_cls.get_special_loading_notes()
    suffix = f" ({notes})" if notes else ""
    return f"Preparing {model_cls.display_name}{suffix}"


def run_training_choice(choice: int, console: Console, results_dir: Path = RESULTS_DIR, publication: bool = False) -> dict[str, dict[str, Any]]:
    spec = model_for_choice(choice)
    if spec is None:
        raise ValueError(f"Unsupported choice: {choice}")

    console.print(f"[yellow]{_model_summary_line(spec.model_cls)}[/yellow]")

    # Load model and print loading notes if any
    model_instance = spec.model_cls()
    notes = model_instance.get_special_loading_notes()
    if notes:
        console.print(f"[dim]Loading notes: {notes}[/dim]")

    if publication:
        from scripts.dataset_enrichment_engine import ENRICHED_PATH
        if not ENRICHED_PATH.exists():
            raise FileNotFoundError(
                f"Enriched dataset not found at {ENRICHED_PATH}. "
                "Run 'python scripts/dataset_enrichment_engine.py' first."
            )
        config = trainer.TrainingConfig(
            model_cls=spec.model_cls,
            dataset_csv=ENRICHED_PATH,
            epochs=10,
            max_rows=None,
            early_stopping_patience=3,
        )
    else:
        from scripts.dataset_enrichment_engine import ENRICHED_PATH
        config = trainer.TrainingConfig(
            model_cls=spec.model_cls,
            dataset_csv=ENRICHED_PATH,
            epochs=1,
            max_rows=802,
            early_stopping_patience=1
        )

    artifacts = trainer.run_training(config)
    metrics = evaluator.run_evaluation(artifacts.model, artifacts.tokenizer, artifacts.test_loader, artifacts.config)
    evaluator.save_metrics(metrics, spec.model_cls.short_name)
    dashboard_exporter.main([])
    return load_metrics_files(results_dir)


def run_sequential_training(console: Console, publication: bool = False) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for spec in MODEL_ZOO:
        results = run_training_choice(spec.choice, console, publication=publication)
    return results


def prompt_choice(input_fn: Callable[[str], str]) -> int:
    return int(input_fn("Select [1-7]: ").strip())


def main(input_fn: Callable[[str], str] = input, console: Console | None = None, publication: bool = False) -> dict[str, dict[str, Any]]:
    console = console or Console()
    if publication:
        console.print("[bold yellow]Running in PUBLICATION MODE (epochs=10, max_rows=None)[/bold yellow]")
    console.print(header_panel())
    choice = prompt_choice(input_fn)

    if choice == 7:
        results = load_metrics_files()
        show_comparison_report(console, results)
        return results
    elif choice == 8:
        dashboard_exporter.main([])
        results_json_path = REPO_ROOT / "dashboard_web" / "data" / "results.json"
        console.print(f"[green]Dashboard data exported to: {results_json_path}[/green]")
        return load_metrics_files()
    elif choice == 6:
        results = run_sequential_training(console, publication=publication)
        show_comparison_report(console, results)
        return results
    elif choice in {1, 2, 3, 4, 5}:
        results = run_training_choice(choice, console, publication=publication)
        show_comparison_report(console, results)
        return results
    else:
        console.print("[red]Invalid choice.[/red]")
        return {}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive MediTriageAI experiment runner.")
    parser.add_argument("--publication", action="store_true", help="Run with full publication configuration.")
    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    main(publication=args.publication)