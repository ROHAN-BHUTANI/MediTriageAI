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


def test_frozen_training_config_no_frozen_instance_error(monkeypatch, tmp_path):
    from dataclasses import FrozenInstanceError
    from scripts.train import TrainingConfig
    from models.xlm_roberta import XLMRobertaLargeModel

    config = TrainingConfig(model_cls=XLMRobertaLargeModel, eval_mode="publication")
    assert config.eval_mode == "publication"

    # Verify that TrainingConfig remains frozen
    with pytest.raises(FrozenInstanceError):
        config.eval_mode = "smoke"

    # Test run_evaluation_only with mocks to guarantee no FrozenInstanceError occurs during evaluation
    from scripts import run_experiment

    checkpoint_dir = tmp_path / "results" / "xlm_roberta_large"
    checkpoint_dir.mkdir(parents=True)
    ckpt_path = checkpoint_dir / "checkpoint.pt"
    ckpt_path.write_bytes(b"dummy")

    class MockConsole:
        def print(self, *args, **kwargs):
            pass

    monkeypatch.setattr(run_experiment, "get_dataset_path", lambda: tmp_path / "dataset.parquet")
    monkeypatch.setattr(run_experiment, "robust_load_checkpoint", lambda *a, **kw: {"model_state_dict": {}})

    class DummyBuiltModel:
        def load_state_dict(self, state_dict):
            pass
        def to(self, device):
            return self

    class DummyModelMeta:
        display_name = "Dummy Model"
        short_name = "dummy_model"
        def get_tokenizer(self):
            return None
        def build(self, path):
            return DummyBuiltModel()
        @classmethod
        def needs_vocab_injection(cls):
            return False

    dummy_spec = run_experiment.ExperimentModel(1, DummyModelMeta)
    monkeypatch.setattr(run_experiment, "_get_model_spec", lambda path: dummy_spec)

    monkeypatch.setattr("src.dataset.load_split_rows", lambda *a, **kw: [{"dummy": 1}])
    monkeypatch.setattr("scripts.train._build_split_loader", lambda *a, **kw: [1])
    monkeypatch.setattr(
        "scripts.evaluate.run_evaluation",
        lambda model, tok, loader, cfg, expected_test_rows=None, eval_mode=None: {
            "n_test_rows": 1,
            "eval_mode": eval_mode,
            "is_full_eval": True,
        },
    )
    monkeypatch.setattr("scripts.evaluate.save_metrics", lambda *a, **kw: None)
    monkeypatch.setattr("scripts.export_dashboard_data.main", lambda *a, **kw: None)

    # Must execute smoothly without FrozenInstanceError
    run_experiment.run_evaluation_only(MockConsole(), ckpt_path, mode="publication")

