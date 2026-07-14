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

    def fake_load_metrics_files(*args, **kwargs):
        return {"xlm_roberta_large": {"model_display_name": "XLM-RoBERTa-large", "specialist_macro_f1": 0.9, "severity_macro_f1": 0.8, "is_novel_contribution": True}}

    def fake_run_training_choice(*args, **kwargs):
        calls["training"] += 1
        return {}

    monkeypatch.setattr(runner, "load_metrics_files", fake_load_metrics_files)
    monkeypatch.setattr(runner, "run_training_choice", fake_run_training_choice)
    console = Console(record=True)
    results = runner.main(input_fn=lambda _: "6", console=console)
    assert results["xlm_roberta_large"]["model_display_name"] == "XLM-RoBERTa-large"
    assert calls["training"] == 0
