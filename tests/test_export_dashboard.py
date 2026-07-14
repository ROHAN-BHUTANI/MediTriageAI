from __future__ import annotations

import json
from pathlib import Path

from scripts import export_dashboard_data as exporter


def test_dataset_statistics_reads_dataset_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "dataset.csv"
    csv_path.write_text(
        "tracking_id,split,language,department_code,severity_label\n"
        "1,train,en,ED,S1\n"
        "2,val,hinglish,GI,S2\n"
        "3,test,en,GI,S2\n",
        encoding="utf-8",
    )
    stats = exporter.dataset_statistics(csv_path)
    assert stats["total_rows"] == 3
    assert stats["train_rows"] == 1
    assert stats["departments"] == 2
    assert "en" in stats["languages"]


def test_build_payload_includes_models_and_summary(tmp_path: Path, monkeypatch) -> None:
    results_dir = tmp_path / "results"
    (results_dir / "xlm_roberta_large").mkdir(parents=True)
    (results_dir / "mbert").mkdir(parents=True)
    (results_dir / "xlm_roberta_large" / "metrics.json").write_text(
        json.dumps({"model_display_name": "XLM-RoBERTa-large", "specialist_macro_f1": 0.9, "severity_macro_f1": 0.8, "is_novel_contribution": True}),
        encoding="utf-8",
    )
    (results_dir / "mbert" / "metrics.json").write_text(
        json.dumps({"model_display_name": "mBERT", "specialist_macro_f1": 0.7, "severity_macro_f1": 0.6}),
        encoding="utf-8",
    )
    csv_path = tmp_path / "dataset.csv"
    csv_path.write_text("tracking_id,split,language,department_code,severity_label\n1,train,en,ED,S1\n", encoding="utf-8")
    payload = exporter.build_payload(results_dir=results_dir, dataset_csv=csv_path)
    assert payload["project"] == "MediTriageAI"
    assert len(payload["models"]) == 2
    assert "0.200" in payload["novelty_summary"]


def test_write_dashboard_json_creates_file(tmp_path: Path) -> None:
    output_path = tmp_path / "dashboard" / "data" / "results.json"
    csv_path = tmp_path / "dataset.csv"
    csv_path.write_text("tracking_id,split,language,department_code,severity_label\n1,train,en,ED,S1\n", encoding="utf-8")
    result_path = exporter.write_dashboard_json(output_path=output_path, results_dir=tmp_path / "results", dataset_csv=csv_path)
    assert result_path.exists()
    data = json.loads(result_path.read_text(encoding="utf-8"))
    assert data["project"] == "MediTriageAI"
