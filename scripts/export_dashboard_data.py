"""Export MediTriageAI experiment results for the dashboard."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def generate_novelty_summary(*args):
    return "Dummy Novelty Summary: 0.200"

RESULTS_DIR = REPO_ROOT / "results"
DASHBOARD_DIR = REPO_ROOT / "dashboard_web"
DASHBOARD_DATA_PATH = DASHBOARD_DIR / "data" / "results.json"
DATASET_CSV = REPO_ROOT / "data" / "processed" / "enriched" / "dataset_enriched.csv"
BUILD_MANIFEST_PATH = REPO_ROOT / "meditriage" / "data" / "processed" / "build_manifest.json"


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_metrics(result_dir: Path = RESULTS_DIR) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    if not result_dir.exists():
        return results
    for metrics_path in sorted(result_dir.glob("*/metrics.json")):
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


def load_build_manifest(manifest_path: Path = BUILD_MANIFEST_PATH) -> dict[str, Any]:
    if not manifest_path.exists():
        return {}
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def dataset_statistics(dataset_csv: Path = DATASET_CSV, manifest_path: Path = BUILD_MANIFEST_PATH) -> dict[str, Any]:
    manifest = load_build_manifest(manifest_path)
    if not dataset_csv.exists():
        return {
            "total_rows": int(manifest.get("n_total_rows", 0)),
            "train_rows": int(manifest.get("split_row_counts", {}).get("train", 0)),
            "val_rows": int(manifest.get("split_row_counts", {}).get("val", 0)),
            "test_rows": int(manifest.get("split_row_counts", {}).get("test", 0)),
            "departments": int(len(manifest.get("department_distribution", {}))),
            "severity_levels": int(len(manifest.get("severity_heuristic_distribution", {}))),
            "languages": list(manifest.get("language_distribution", {}).keys()),
        }

    df = pd.read_csv(dataset_csv)
    return {
        "total_rows": int(len(df)),
        "train_rows": int((df["split"] == "train").sum()),
        "val_rows": int((df["split"] == "val").sum()),
        "test_rows": int((df["split"] == "test").sum()),
        "departments": int(df.get("department_code", pd.Series(dtype=str)).dropna().nunique()),
        "severity_levels": int(
            df.get("severity_label", df.get("severity_heuristic", pd.Series(dtype=str))).dropna().nunique()
        ),
        "languages": df.get("language", pd.Series(dtype=str)).dropna().unique().tolist(),
    }


def to_dashboard_model(slug: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": metrics.get("model_display_name", slug),
        "specialist_f1": float(metrics.get("specialist_macro_f1", metrics.get("specialist_f1", 0.0))),
        "severity_f1": float(metrics.get("severity_macro_f1", metrics.get("severity_f1", 0.0))),
        "is_novel": bool(metrics.get("is_novel_contribution", False)),
        "severity_confusion_matrix": metrics.get("severity_confusion_matrix", []),
    }


def build_payload(
    results_dir: Path = RESULTS_DIR,
    dataset_csv: Path = DATASET_CSV,
    manifest_path: Path = BUILD_MANIFEST_PATH,
) -> dict[str, Any]:
    raw_results = load_metrics(results_dir)
    models = sorted(
        [to_dashboard_model(slug, metrics) for slug, metrics in raw_results.items()],
        key=lambda item: (not item["is_novel"], -item["specialist_f1"], item["name"].lower()),
    )
    novelty_input = {
        slug: {
            "model_display_name": metrics.get("model_display_name", slug),
            "is_novel_contribution": bool(metrics.get("is_novel_contribution", False)),
            "specialist_macro_f1": float(metrics.get("specialist_macro_f1", metrics.get("specialist_f1", 0.0))),
            "severity_macro_f1": float(metrics.get("severity_macro_f1", metrics.get("severity_f1", 0.0))),
        }
        for slug, metrics in raw_results.items()
    }
    return {
        "last_updated": now_utc(),
        "project": "MediTriageAI",
        "models": models,
        "dataset_stats": dataset_statistics(dataset_csv, manifest_path=manifest_path),
        "novelty_summary": generate_novelty_summary(novelty_input),
    }


def write_dashboard_json(
    output_path: Path = DASHBOARD_DATA_PATH,
    results_dir: Path = RESULTS_DIR,
    dataset_csv: Path = DATASET_CSV,
    manifest_path: Path = BUILD_MANIFEST_PATH,
) -> Path:
    payload = build_payload(results_dir=results_dir, dataset_csv=dataset_csv, manifest_path=manifest_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export MediTriageAI results to dashboard_web/data/results.json.")
    parser.add_argument("--dry-run", action="store_true", help="Validate the dashboard payload without writing files.")
    parser.add_argument(
        "--results-dir", type=Path, default=RESULTS_DIR, help="Directory containing results/*/metrics.json files."
    )
    parser.add_argument(
        "--dataset-csv", type=Path, default=DATASET_CSV, help="Processed dataset CSV used for summary statistics."
    )
    parser.add_argument(
        "--manifest", type=Path, default=BUILD_MANIFEST_PATH, help="Path to build_manifest.json."
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    payload = build_payload(
        results_dir=args.results_dir, dataset_csv=args.dataset_csv, manifest_path=args.manifest
    )
    if args.dry_run:
        # Validate by trying to serialize to JSON
        json.dumps(payload)
        print("Validation passed")
    else:
        write_dashboard_json(
            output_path=DASHBOARD_DATA_PATH,
            results_dir=args.results_dir,
            dataset_csv=args.dataset_csv,
            manifest_path=args.manifest,
        )
        print(f"Dashboard data written to {DASHBOARD_DATA_PATH}")


if __name__ == "__main__":
    main()