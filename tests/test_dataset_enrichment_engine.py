# tests for the Dataset Enrichment Engine

import json
import os
import pandas as pd
import pytest
from pathlib import Path

# Import functions from the enrichment engine script
from scripts.dataset_enrichment_engine import deterministic_id, main, SEED, THRESHOLDS

# Import registry utilities for additional unit checks
from src.registry import load_config, load_plugins, CONFIG_PATH

@pytest.fixture
def minimal_dataset(tmp_path: Path) -> Path:
    """Create a minimal original dataset required by the enrichment engine.

    The engine expects the file at ``data/processed/improved/dataset_improved.csv``
    relative to the current working directory. This fixture creates the required
    directory structure inside ``tmp_path`` and writes a tiny CSV with the
    minimal columns needed for plugin execution.
    """
    # Directory layout
    improved_dir = tmp_path / "data" / "processed" / "improved"
    improved_dir.mkdir(parents=True, exist_ok=True)

    # Minimal dataset – two rows to keep the test fast
    df = pd.DataFrame([
        {
            "tracking_id": "orig1",
            "seed_id": "0",
            "variant_index": 0,
            "is_perturbed": False,
            "text": "SUBJECTIVE: Patient reports severe abdominal pain and vomiting since yesterday.",
            "department_code": "GI",
            "split": "train",
        },
        {
            "tracking_id": "orig2",
            "seed_id": "0",
            "variant_index": 0,
            "is_perturbed": False,
            "text": "SUBJECTIVE: Patient has cough and fever – high temperature.",
            "department_code": "GEN_MED",
            "split": "train",
        },
    ])
    csv_path = improved_dir / "dataset_improved.csv"
    df.to_csv(csv_path, index=False)
    return csv_path

def test_deterministic_id_format():
    sid = deterministic_id(parent_id="parent123", specialty="cardio", seq=7)
    assert sid == "SYN_CARDIO_parent123_0007"

def test_load_config_contains_expected_keys():
    cfg = load_config()
    assert "seed" in cfg
    assert isinstance(cfg.get("enabled_plugins", []), list)
    thresholds = cfg.get("diversity_thresholds", {})
    for key in ["lexical_diversity", "edit_distance_ratio", "token_overlap", "novelty_score"]:
        assert key in thresholds

def test_load_plugins_returns_instances(minimal_dataset: Path, monkeypatch):
    # Change cwd to project root under tmp_path
    monkeypatch.chdir(minimal_dataset.parent.parent.parent.parent)
    plugins = load_plugins()
    from src.transformation_base import TransformationPlugin
    for p in plugins:
        assert isinstance(p, TransformationPlugin)
    cfg = load_config()
    assert len(plugins) == len(cfg.get("enabled_plugins", []))

def test_full_enrichment_pipeline(minimal_dataset: Path, monkeypatch):
    import scripts.dataset_enrichment_engine as engine
    enriched_dir = minimal_dataset.parent.parent / "enriched"
    enriched_dir.mkdir(parents=True, exist_ok=True)
    
    # Monkeypatch the absolute paths in the engine to use the tmp_path fixture
    monkeypatch.setattr(engine, "ORIG_PATH", minimal_dataset)
    monkeypatch.setattr(engine, "ENRICHED_DIR", enriched_dir)
    monkeypatch.setattr(engine, "SYNTHETIC_PATH", enriched_dir / "synthetic_samples.csv")
    monkeypatch.setattr(engine, "ENRICHED_PATH", enriched_dir / "dataset_enriched.csv")
    monkeypatch.setattr(engine, "DIVERSITY_REPORT", enriched_dir / "synthetic_diversity_report.csv")
    monkeypatch.setattr(engine, "CLINICAL_REPORT", enriched_dir / "clinical_validation_report.csv")
    monkeypatch.setattr(engine, "DUPLICATE_REPORT", enriched_dir / "duplicate_validation_report.csv")
    monkeypatch.setattr(engine, "GEN_STATS", enriched_dir / "generation_statistics.json")
    monkeypatch.setattr(engine, "MANIFEST", enriched_dir / "enrichment_manifest.json")
    monkeypatch.setattr(engine, "REPORT_MD", enriched_dir / "dataset_enrichment_report.md")

    project_root = minimal_dataset.parent.parent.parent.parent
    monkeypatch.chdir(project_root)
    engine.main(dry_run=False)
    expected_files = [
        "synthetic_samples.csv",
        "dataset_enriched.csv",
        "synthetic_diversity_report.csv",
        "clinical_validation_report.csv",
        "duplicate_validation_report.csv",
        "generation_statistics.json",
        "enrichment_manifest.json",
        "dataset_enrichment_report.md",
    ]
    for fname in expected_files:
        assert (enriched_dir / fname).exists(), f"{fname} missing"
    synth_df = pd.read_csv(enriched_dir / "synthetic_samples.csv")
    for col in ["tracking_id", "text", "department_code", "provenance", "passed_diversity"]:
        assert col in synth_df.columns
    assert not synth_df.empty
    enriched_df = pd.read_csv(enriched_dir / "dataset_enriched.csv")
    orig_df = pd.read_csv(minimal_dataset)
    assert len(enriched_df) == len(orig_df) + len(synth_df)
    div_report = pd.read_csv(enriched_dir / "synthetic_diversity_report.csv")
    assert "passed" in div_report.columns
    assert set(div_report["passed"].unique()).issubset({True, False})
    with open(enriched_dir / "generation_statistics.json", "r", encoding="utf-8") as f:
        stats = json.load(f)
    for key in ["seed", "total_original", "total_synthetic", "passed_diversity", "failed_diversity"]:
        assert key in stats
    with open(enriched_dir / "enrichment_manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert isinstance(manifest.get("plugins", []), list)
    assert manifest.get("seed") == SEED
    clinical_df = pd.read_csv(enriched_dir / "clinical_validation_report.csv")
    assert set(["record_id", "passed", "reasons"]).issubset(set(clinical_df.columns))
    dup_df = pd.read_csv(enriched_dir / "duplicate_validation_report.csv")
    assert set(["record_id", "duplicate_type", "source_dataset", "matched_record_id", "reason"]).issubset(set(dup_df.columns))
