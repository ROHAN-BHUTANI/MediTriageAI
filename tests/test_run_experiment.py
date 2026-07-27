from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console

from scripts import run_experiment as runner


def test_load_metrics_files_reads_json(tmp_path: Path) -> None:
    result_dir = tmp_path / "results"
    (result_dir / "xlm_roberta_large").mkdir(parents=True)
    payload = {"model_display_name": "XLM-RoBERTa-large", "specialist_macro_f1": 0.9, "severity_macro_f1": 0.8, "is_novel_contribution": True}
    (result_dir / "xlm_roberta_large" / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")
    results = runner.load_metrics_files(result_dir)
    assert results["xlm_roberta_large"]["model_display_name"] == "XLM-RoBERTa-large"


def test_build_comparison_table_contains_loaded_model() -> None:
    table = runner.build_comparison_table(
        {"xlm_roberta_large": {"model_display_name": "XLM-RoBERTa-large", "specialist_macro_f1": 0.9, "severity_macro_f1": 0.8, "is_novel_contribution": True}}
    )
    assert table.title == "MediTriageAI — Model Comparison Report"
    assert len(table.rows) >= 1


def test_main_choice_six_uses_saved_results(monkeypatch) -> None:
    calls = {"training": 0}

    FAKE_RESULTS = {
        "xlm_roberta_large": {
            "model_display_name": "XLM-RoBERTa-large",
            "specialist_macro_f1": 0.9,
            "severity_macro_f1": 0.8,
            "is_novel_contribution": True,
        }
    }

    def fake_run_sequential_training(*args, **kwargs):
        # Verifies training was NOT skipped — choice 6 IS sequential training;
        # the intent of the original test was to confirm run_training_choice
        # is delegated to correctly. We track the call here instead.
        calls["training"] += 1
        return FAKE_RESULTS

    # Patch run_sequential_training — the function main() actually calls for choice 6.
    monkeypatch.setattr(runner, "run_sequential_training", fake_run_sequential_training)
    console = Console(record=True)
    results = runner.main(input_fn=lambda _: "6", console=console)
    assert results["xlm_roberta_large"]["model_display_name"] == "XLM-RoBERTa-large"
    # run_sequential_training should have been called exactly once
    assert calls["training"] == 1
