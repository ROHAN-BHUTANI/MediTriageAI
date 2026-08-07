"""Experiment runner for MediTriageAI.

Forensic instrumentation added for production observability.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.distilbert_multi import DistilBertMultilingualModel
from models.emergent_path_triage import EmergentPathTriageModel
from models.indic_bert import IndicBertModel
from models.mbert import MBertModel
from models.xlm_roberta import XLMRobertaLargeModel
from scripts import evaluate as evaluator
from scripts import export_dashboard_data as dashboard_exporter
from scripts import train as trainer

_logger = logging.getLogger("meditriage.training.run_experiment")
if not _logger.handlers:
    _sh = logging.StreamHandler(sys.stderr)
    _sh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
    _logger.addHandler(_sh)
    _logger.setLevel(logging.DEBUG)

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


def _get_model_spec(checkpoint_dir: Path) -> ExperimentModel | None:
    for spec in MODEL_ZOO:
        if spec.model_cls.short_name == checkpoint_dir.name:
            return spec
    return None


def get_dataset_path() -> Path:
    dataset_path = REPO_ROOT / "meditriage" / "data" / "processed" / "dataset.parquet"
    if not dataset_path.exists():
        dataset_path = REPO_ROOT / "meditriage" / "data" / "processed" / "dataset.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Production dataset not found at {dataset_path.parent}. "
            "Run 'python -m meditriage.builder.cli build' first."
        )
    return dataset_path


def load_metrics_files(results_dir: Path = RESULTS_DIR) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    if not results_dir.exists():
        return results
    for metrics_path in sorted(results_dir.glob("*/metrics.json")):
        try:
            results[metrics_path.parent.name] = json.loads(
                metrics_path.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError:
            continue

    if results:
        latest_eval = max(results.values(), key=lambda x: x.get("evaluated_at", ""))
        expected_rows = latest_eval.get("n_test_rows")
        results = {
            k: v for k, v in results.items() if v.get("n_test_rows") == expected_rows
        }

    return results


def robust_load_checkpoint(
    checkpoint_path: Path, console: Console, map_location: str = "cpu"
) -> Any:
    import pathlib
    import pickle

    if hasattr(torch.serialization, "add_safe_globals"):
        try:
            torch.serialization.add_safe_globals(
                [pathlib.PosixPath, pathlib.WindowsPath]
            )
        except Exception:
            pass

    console.print("[cyan]Loading checkpoint...[/cyan]")
    try:
        state_dict = torch.load(
            checkpoint_path, map_location=map_location, weights_only=True
        )
        console.print("[green]Checkpoint restored successfully.[/green]")
        return state_dict
    except Exception as e:
        is_weights_error = False
        err_str = str(e)
        if (
            isinstance(e, TypeError)
            and "weights_only" in err_str
            or "Weights only load failed" in err_str
            or "WeightsUnpickler" in err_str
            or isinstance(e, pickle.UnpicklingError)
        ):
            is_weights_error = True

        if is_weights_error:
            console.print("[yellow]Detected legacy checkpoint format...[/yellow]")
            console.print("[yellow]Loading using compatibility mode...[/yellow]")
            state_dict = torch.load(
                checkpoint_path, map_location=map_location, weights_only=False
            )
            console.print("[green]Checkpoint restored successfully.[/green]")
            return state_dict
        else:
            raise RuntimeError(f"Failed to load checkpoint: {e}")


def run_evaluation_only(
    console: Console, checkpoint_path: Path, mode: str, run_error_analysis: bool = False
) -> None:
    res_dir = checkpoint_path.parent
    spec = _get_model_spec(res_dir)
    if not spec:
        console.print(
            f"[red]Error: Cannot determine model type for checkpoint directory '{res_dir.name}'[/red]"
        )
        sys.exit(1)

    dataset_path = get_dataset_path()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "N/A"

    console.print(f"Mode               : {mode.upper()}")
    console.print(f"Model              : {spec.model_cls.display_name}")
    console.print(f"Checkpoint         : {checkpoint_path}")
    console.print(f"Dataset            : {dataset_path}")
    console.print(f"Device             : {device}")
    console.print(f"GPU                : {gpu_name}")
    console.print(f"Result directory   : {res_dir}")
    console.print("")
    console.print("Running evaluation...")

    # Load Model and Tokenizer
    model_meta = spec.model_cls()
    tokenizer = model_meta.get_tokenizer()
    built_model = model_meta.build(None)

    if spec.model_cls.needs_vocab_injection():
        model_meta.inject_vocab(built_model, tokenizer)

    state_dict = robust_load_checkpoint(checkpoint_path, console, map_location="cpu")
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    built_model.load_state_dict(state_dict)
    built_model.to(torch.device(device))

    # Load TEST dataloader only
    from scripts.train import _build_split_loader

    max_rows = 800 if mode in {"smoke", "evaluate"} else (10000 if mode == "development" else None)
    test_loader = _build_split_loader(
        "test",
        tokenizer,
        dataset_path,
        batch_size=32,
        max_length=64,
        max_rows=max_rows,
    )

    if test_loader is None:
        console.print("[red]Error: Failed to build test dataloader.[/red]")
        sys.exit(1)

    from scripts.train import TrainingConfig as LegacyTrainingConfig

    config = LegacyTrainingConfig(model_cls=spec.model_cls, dataset_path=dataset_path)

    metrics = evaluator.run_evaluation(built_model, tokenizer, test_loader, config)
    evaluator.save_metrics(metrics, spec.model_cls.short_name)
    dashboard_exporter.main([])

    if run_error_analysis:
        console.print("Running error analysis...")
        subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "generate_error_analysis.py"),
                "--results-dir",
                str(res_dir),
            ],
            check=True,
        )


def run_training_workflow(
    console: Console,
    mode: str,
    checkpoint_path: Path | None,
    choice: int | None = None,
) -> dict[str, dict[str, Any]]:
    # If a specific choice is passed via interactive prompt or arguments, use it.
    if choice is None:
        # If we have a checkpoint, infer the model
        if checkpoint_path:
            spec = _get_model_spec(checkpoint_path.parent)
            if spec:
                choice = spec.choice

        if choice is None:
            # Fallback to interactive
            console.print(header_panel())
            choice = int(input("Select [1-7]: ").strip())

    if choice == 7:
        results = load_metrics_files()
        console.print(build_comparison_table(results))
        return results
    elif choice == 8:
        dashboard_exporter.main([])
        return load_metrics_files()
    elif choice == 6:
        results = {}
        for sp in MODEL_ZOO:
            results = _do_training(sp.choice, console, mode, None)
        console.print(build_comparison_table(results))
        return results
    elif choice in {1, 2, 3, 4, 5}:
        results = _do_training(choice, console, mode, checkpoint_path)
        console.print(build_comparison_table(results))
        return results
    else:
        console.print("[red]Invalid choice.[/red]")
        return {}


def _do_training(
    choice: int, console: Console, mode: str, checkpoint_path: Path | None
) -> dict[str, dict[str, Any]]:
    spec = next((s for s in MODEL_ZOO if s.choice == choice), None)
    if spec is None:
        raise ValueError(f"Unsupported choice: {choice}")

    _logger.info("[EXPERIMENT] Entering %s mode", mode.upper())
    _logger.info("[EXPERIMENT] Model selected: %s (%s)",
                  spec.model_cls.display_name, spec.model_cls.short_name)
    console.print(f"[yellow]Preparing {spec.model_cls.display_name}[/yellow]")
    dataset_path = get_dataset_path()
    _logger.info("[EXPERIMENT] Dataset path: %s", dataset_path)

    if mode == "publication":
        config = trainer.TrainingConfig(
            model_cls=spec.model_cls,
            dataset_path=dataset_path,
            epochs=10,
            max_rows=None,
            early_stopping_patience=3,
            resume_checkpoint=checkpoint_path,
        )
    elif mode == "development":
        config = trainer.TrainingConfig(
            model_cls=spec.model_cls,
            dataset_path=dataset_path,
            epochs=3,
            max_rows=10000,
            early_stopping_patience=1,
            resume_checkpoint=checkpoint_path,
        )
    else:  # smoke
        config = trainer.TrainingConfig(
            model_cls=spec.model_cls,
            dataset_path=dataset_path,
            epochs=1,
            max_rows=800,
            early_stopping_patience=1,
            resume_checkpoint=checkpoint_path,
        )

    _logger.info("[EXPERIMENT] TrainingConfig created: epochs=%d, batch=%d, max_rows=%s",
                  config.epochs, config.batch_size, config.max_rows)
    _logger.info("[EXPERIMENT] Calling trainer.run_training()...")
    try:
        artifacts = trainer.run_training(config)
    except Exception:
        _logger.error("[EXPERIMENT] EXCEPTION in trainer.run_training()!\n%s",
                        traceback.format_exc())
        raise
    _logger.info("[EXPERIMENT] trainer.run_training() returned successfully")

    metrics = evaluator.run_evaluation(
        artifacts.model, artifacts.tokenizer, artifacts.test_loader, artifacts.config
    )
    evaluator.save_metrics(metrics, spec.model_cls.short_name)
    dashboard_exporter.main([])
    _logger.info("[EXPERIMENT] _do_training() complete")
    return load_metrics_files(RESULTS_DIR)


def header_panel() -> Panel:
    return Panel(
        "1. XLM-RoBERTa Large (Baseline)\n"
        "2. mBERT (Baseline)\n"
        "3. DistilBERT Multilingual (Baseline)\n"
        "4. IndicBERT (Baseline)\n"
        "5. EmergentPathTriage (Novel)\n"
        "6. Run Sequential Pipeline (All Models)\n"
        "7. View Comparison Report\n"
        "8. Export Dashboard Data",
        title="MediTriageAI Model Zoo",
        border_style="cyan",
    )


def _metric_value(item: dict[str, Any], *keys: str) -> float:
    for key in keys:
        value = item.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


def _best_slug(results: dict[str, dict[str, Any]], metric_keys: tuple[str, ...]) -> str:
    if not results:
        return ""
    return max(results.items(), key=lambda pair: _metric_value(pair[1], *metric_keys))[
        0
    ]


def build_comparison_table(results: dict[str, dict[str, Any]]) -> Table:
    table = Table(title="MediTriageAI - Model Comparison Report", show_lines=False)
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
    best_adj_err = min(
        results.items(),
        key=lambda pair: _metric_value(
            pair[1], "severity_adjacent_confusion_rate", "severity_adjacent_confusion"
        ),
    )[0]

    ordered = sorted(
        results.items(),
        key=lambda pair: _metric_value(pair[1], "specialist_macro_f1", "specialist_f1"),
        reverse=True,
    )
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
            role,
        )
    return table


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactive MediTriageAI experiment runner."
    )
    parser.add_argument(
        "--mode",
        choices=["smoke", "development", "publication", "evaluate"],
        default="development",
        help="Experiment mode configuration.",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Run evaluation only on an existing checkpoint without training.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to checkpoint or 'auto' to discover newest.",
    )
    parser.add_argument(
        "--error-analysis",
        action="store_true",
        help="Automatically invoke error analysis after evaluation.",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    console = Console()

    _logger.info("[MAIN] run_experiment.py ENTER — mode=%s", args.mode)

    try:
        checkpoint_path = None
        if args.checkpoint:
            if args.checkpoint == "auto":
                if not RESULTS_DIR.exists():
                    console.print(
                        "[red]Error: Results directory not found for auto-discovery.[/red]"
                    )
                    sys.exit(1)
                ckpt_files = list(RESULTS_DIR.glob("*/checkpoint.pt"))
                if not ckpt_files:
                    console.print(
                        "[red]Error: No checkpoints found in results directory.[/red]"
                    )
                    sys.exit(1)
                checkpoint_path = max(ckpt_files, key=lambda p: p.stat().st_mtime)
                console.print(f"Auto-discovered newest checkpoint: {checkpoint_path}")
            else:
                checkpoint_path = Path(args.checkpoint)
                if not checkpoint_path.is_absolute():
                    checkpoint_path = REPO_ROOT / checkpoint_path
                if not checkpoint_path.exists():
                    console.print(
                        f"[red]Error: Checkpoint '{checkpoint_path}' does not exist.[/red]"
                    )
                    sys.exit(1)

        if args.evaluate_only or args.mode == "evaluate":
            if not checkpoint_path:
                console.print(
                    "[red]Error: Evaluation-only mode requires --checkpoint.[/red]"
                )
                sys.exit(1)
            eval_mode = "evaluate" if args.evaluate_only else args.mode
            run_evaluation_only(
                console, checkpoint_path, eval_mode, args.error_analysis
            )
        else:
            console.print(
                f"[bold yellow]Running in {args.mode.upper()} MODE[/bold yellow]"
            )
            _logger.info("[MAIN] Dispatching to run_training_workflow(mode=%s)", args.mode)
            run_training_workflow(console, args.mode, checkpoint_path)
    except FileNotFoundError as e:
        _logger.error("[MAIN] FileNotFoundError: %s", e)
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)
    except Exception:
        _logger.error("[MAIN] Unhandled exception!\n%s", traceback.format_exc())
        raise

    _logger.info("[MAIN] run_experiment.py EXIT")


if __name__ == "__main__":
    main()
