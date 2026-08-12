from pathlib import Path

from meditriage.builder.config import Config
from meditriage.builder.orchestrator import Builder


def test_builder_end_to_end(tmp_path):
    config_dict = {
        "random_seed": 42,
        "splits": {"train": 0.8, "val": 0.1, "test": 0.1},
        "active_datasets": ["mtsamples"],
        "augmentation": {
            "hinglish": {
                "enabled_for": ["mtsamples"],
                "variants_per_seed": 1,
                "substitution_rate": 0.5,
            }
        },
        "deduplication": {"strategy": "exact_match", "priority_order": ["mtsamples"]},
    }

    config = Config(config_dict, raw_yaml="")
    # Use real base_dir to find raw datasets
    base_dir = Path(__file__).resolve().parent.parent.parent

    # We will override out_dir to be tmp_path
    builder = Builder(config, base_dir)
    builder.out_dir = tmp_path
    builder.processed_dir = tmp_path / "processed"
    builder.build_dir = tmp_path / "build_temp"
    builder.processed_dir.mkdir(parents=True, exist_ok=True)
    builder.build_dir.mkdir(parents=True, exist_ok=True)

    builder.build(force=True)

    assert (tmp_path / "processed" / "dataset.csv").exists()
    assert (tmp_path / "processed" / "build_manifest.json").exists()
    assert (tmp_path / "processed" / "dataset_statistics.json").exists()


def test_builder_force_false_raises_file_exists_error(tmp_path):
    """Test A: Verify builder raises FileExistsError if dataset.csv exists and force=False."""
    import pytest

    config = Config({"active_datasets": ["mtsamples"]}, raw_yaml="")
    base_dir = Path(__file__).resolve().parent.parent.parent

    builder = Builder(config, base_dir)
    builder.processed_dir = tmp_path / "processed"
    builder.processed_dir.mkdir(parents=True, exist_ok=True)

    # Create dummy dataset.csv
    (builder.processed_dir / "dataset.csv").write_text("old_header\nold_data\n")

    with pytest.raises(FileExistsError, match="Dataset exists. Use --force to overwrite"):
        builder.build(force=False)


def test_builder_force_true_prevents_csv_append_corruption(tmp_path):
    """Test B-E: Verify force=True unlinks old CSV so resulting CSV contains ONLY new build with single header."""
    import json
    import pandas as pd

    base_dir = Path(__file__).resolve().parent.parent.parent

    # Create mock raw dataset with 5 records
    mock_raw_dir = tmp_path / "datasets" / "raw"
    mock_meddialog_dir = mock_raw_dir / "meddialog_en"
    mock_meddialog_dir.mkdir(parents=True, exist_ok=True)

    records = [
        {"patient_message": f"Question {i}", "doctor_response": f"Answer {i}"}
        for i in range(5)
    ]
    with open(mock_meddialog_dir / "train.json", "w", encoding="utf-8") as f:
        json.dump(records, f)

    config_dict = {
        "active_datasets": ["meddialog_en"],
        "splits": {"train": 0.8, "val": 0.1, "test": 0.1},
        "deduplication": {"priority_order": ["meddialog_en"], "strategy": "exact_match"},
    }
    config = Config(config_dict, raw_yaml="")

    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Write a stale dataset.csv with 1,000 old dummy rows to simulate old dataset
    stale_csv = processed_dir / "dataset.csv"
    stale_lines = ["id,split,dataset_source,language,raw_text,department,triage_level\n"] + [
        f"old::{i},train,old_src,en,old text {i},GEN_MED,S3\n" for i in range(1000)
    ]
    stale_csv.write_text("".join(stale_lines))

    stale_pq = processed_dir / "dataset.parquet"
    pd.DataFrame({"id": ["old"]}).to_parquet(stale_pq)

    # Run builder with force=True
    builder = Builder(config, base_dir)
    builder.raw_dir = mock_raw_dir
    builder.processed_dir = processed_dir
    builder.build_dir = tmp_path / "build_temp"

    builder.build(force=True)

    # Verify CSV was cleanly unlinked and rewritten with ONLY new 5 records
    df_csv = pd.read_csv(processed_dir / "dataset.csv")
    df_pq = pd.read_parquet(processed_dir / "dataset.parquet")

    # Assert row count is 5 (NOT 1005 from stale appending)
    assert len(df_csv) == 5, f"Expected 5 rows in CSV, got {len(df_csv)} (stale appending occurred!)"
    assert len(df_pq) == 5, f"Expected 5 rows in Parquet, got {len(df_pq)}"
    assert len(df_csv) == len(df_pq), "CSV row count does not equal Parquet row count"

    # Assert no stale IDs remain
    assert not any("old::" in str(rid) for rid in df_csv["id"]), "Stale old records were found in CSV!"

    # Assert CSV header is not duplicated
    raw_csv_text = (processed_dir / "dataset.csv").read_text()
    header_count = raw_csv_text.count("id,split,dataset_source")
    assert header_count == 1, f"Expected exactly 1 CSV header line, found {header_count}"
