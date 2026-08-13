"""Regression and integrity tests for MediTriageAI evaluation pipeline."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from scripts.run_experiment import build_arg_parser, resolve_evaluation_policy
from scripts.evaluate import run_evaluation, save_metrics
from scripts import export_dashboard_data as dashboard_exporter


def test_resolve_evaluation_policy():
    assert resolve_evaluation_policy("smoke") == ("smoke", 800)
    assert resolve_evaluation_policy("development") == ("development", 10000)
    assert resolve_evaluation_policy("dev") == ("development", 10000)
    assert resolve_evaluation_policy("publication") == ("publication", None)
    assert resolve_evaluation_policy("evaluate") == ("publication", None)
    assert resolve_evaluation_policy(None) == ("publication", None)

    with pytest.raises(ValueError, match="Unknown or ambiguous evaluation mode"):
        resolve_evaluation_policy("invalid_mode")


def test_arg_parser_evaluation_defaults():
    parser = build_arg_parser()

    # --evaluate-only without --mode should default mode to None (resolving to publication)
    args = parser.parse_args(["--evaluate-only"])
    assert args.evaluate_only is True
    assert args.mode is None
    assert resolve_evaluation_policy(args.mode) == ("publication", None)

    # --evaluate-only with --mode smoke
    args_smoke = parser.parse_args(["--evaluate-only", "--mode", "smoke"])
    assert args_smoke.mode == "smoke"
    assert resolve_evaluation_policy(args_smoke.mode) == ("smoke", 800)

    # --evaluate-only with --mode development
    args_dev = parser.parse_args(["--evaluate-only", "--mode", "development"])
    assert args_dev.mode == "development"
    assert resolve_evaluation_policy(args_dev.mode) == ("development", 10000)

    # --evaluate-only with --mode publication
    args_pub = parser.parse_args(["--evaluate-only", "--mode", "publication"])
    assert args_pub.mode == "publication"
    assert resolve_evaluation_policy(args_pub.mode) == ("publication", None)

    # --mode evaluate
    args_eval = parser.parse_args(["--mode", "evaluate"])
    assert args_eval.mode == "evaluate"
    assert resolve_evaluation_policy(args_eval.mode) == ("publication", None)


def test_evaluation_integrity_assertion_fails_on_truncated_count():
    class DummyConfig:
        model_display_name = "DummyModel"
        model_short_name = "dummy_model"
        is_novel_contribution = False
        max_rows = 800
        eval_mode = "publication"

    metrics = {
        "model_display_name": "DummyModel",
        "model_short_name": "dummy_model",
        "eval_mode": "publication",
        "is_full_eval": True,
        "max_rows": 800,
        "n_test_rows": 800,
        "expected_test_rows": 778991,
    }

    # saving metrics with max_rows=800 when eval_mode="publication" MUST raise ValueError
    with pytest.raises(ValueError, match="CRITICAL EVALUATION INTEGRITY FAILURE"):
        save_metrics(metrics, "dummy_model")


def test_evaluation_integrity_assertion_fails_on_mismatched_test_count():
    metrics = {
        "model_display_name": "DummyModel",
        "model_short_name": "dummy_model",
        "eval_mode": "publication",
        "is_full_eval": True,
        "max_rows": None,
        "n_test_rows": 800,
        "expected_test_rows": 778991,
    }

    # n_test_rows=800 != expected_test_rows=778991 MUST raise ValueError
    with pytest.raises(ValueError, match="CRITICAL EVALUATION INTEGRITY FAILURE"):
        save_metrics(metrics, "dummy_model")


def test_800_row_evaluation_cannot_be_mistaken_for_publication(tmp_path: Path):
    results_dir = tmp_path / "results"
    (results_dir / "smoke_model").mkdir(parents=True)
    (results_dir / "pub_model").mkdir(parents=True)

    # Smoke evaluation file (partial)
    (results_dir / "smoke_model" / "metrics.json").write_text(
        json.dumps(
            {
                "model_display_name": "Smoke Model",
                "eval_mode": "smoke",
                "is_full_eval": False,
                "max_rows": 800,
                "n_test_rows": 800,
                "specialist_macro_f1": 0.5,
                "severity_macro_f1": 0.5,
            }
        ),
        encoding="utf-8",
    )

    # Publication evaluation file (full)
    (results_dir / "pub_model" / "metrics.json").write_text(
        json.dumps(
            {
                "model_display_name": "Publication Model",
                "eval_mode": "publication",
                "is_full_eval": True,
                "max_rows": None,
                "n_test_rows": 778991,
                "specialist_macro_f1": 0.9,
                "severity_macro_f1": 0.9,
            }
        ),
        encoding="utf-8",
    )

    loaded = dashboard_exporter.load_metrics(results_dir)

    # Smoke model MUST be excluded from publication dashboard load
    assert "smoke_model" not in loaded
    assert "pub_model" in loaded
    assert loaded["pub_model"]["n_test_rows"] == 778991
