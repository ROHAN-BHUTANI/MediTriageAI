"""Tests for dataset acquisition governance and allowlist filtering."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "datasets"))

from download_hf import DATASET_SPECS, acquire_all_datasets
from meditriage.builder.config import Config
from unittest.mock import MagicMock, patch


def test_active_dataset_count():
    """Verify production dataset_config.yaml declares exactly 10 active datasets."""
    config = Config.from_yaml(PROJECT_ROOT / "config" / "dataset_config.yaml")
    assert len(config.active_datasets) == 10
    excluded = {"medqa_usmle", "medical_meadow_medqa", "l3cube_code_mixed"}
    assert not any(ds in config.active_datasets for ds in excluded)


def test_dataset_specs_registry_count():
    """Verify DATASET_SPECS contains 13 total known specs including historical definitions."""
    assert len(DATASET_SPECS) == 13
    spec_names = {spec[0] for spec in DATASET_SPECS}
    assert "medqa_usmle" in spec_names
    assert "medical_meadow_medqa" in spec_names
    assert "l3cube_code_mixed" in spec_names


def test_active_spec_completeness():
    """Verify every configured active dataset has an acquisition spec in DATASET_SPECS."""
    config = Config.from_yaml(PROJECT_ROOT / "config" / "dataset_config.yaml")
    spec_names = {spec[0] for spec in DATASET_SPECS}
    for ds in config.active_datasets:
        assert ds in spec_names, f"Active dataset {ds} missing acquisition spec in DATASET_SPECS"


def test_standalone_download_filtering():
    """Verify acquire_all_datasets filters specs down to exactly the 10 active datasets."""
    config = Config.from_yaml(PROJECT_ROOT / "config" / "dataset_config.yaml")
    active_set = set(config.active_datasets)

    with patch("download_hf.acquire_single_dataset") as mock_acquire:
        mock_acquire.side_effect = lambda spec: (spec[0], "SKIPPED_EXISTS", 1, 100, "test")
        results = acquire_all_datasets()
        acquired_names = [res[0] for res in results]

        assert len(acquired_names) == 10
        assert set(acquired_names) == active_set
        assert "medqa_usmle" not in acquired_names
        assert "medical_meadow_medqa" not in acquired_names
        assert "l3cube_code_mixed" not in acquired_names


def test_bootstrap_extraction_governance():
    """Verify extract_all_archives respects active_set and skips excluded datasets."""
    from bootstrap import extract_all_archives

    config = Config.from_yaml(PROJECT_ROOT / "config" / "dataset_config.yaml")
    active_set = set(config.active_datasets)

    with patch("bootstrap.extract_archive") as mock_extract:
        # Test with active set (excluding medqa_usmle and l3cube_code_mixed)
        with patch("bootstrap.RAW") as mock_raw:
            # Setup mock zip files
            zip_mock = MagicMock()
            zip_mock.exists.return_value = True
            target_mock = MagicMock()
            target_mock.exists.return_value = False

            def raw_div(name):
                d = MagicMock()
                if name == "nhamcs_ed":
                    d.__getitem__.return_value = zip_mock
                    d.__truediv__.return_value = target_mock
                else:
                    d.__getitem__.return_value = zip_mock
                    d.__truediv__.return_value = target_mock
                return d

            mock_raw.__truediv__.side_effect = raw_div

            extract_all_archives(active_set)
            # extract_archive should NOT be called for medqa_usmle or l3cube_code_mixed
            for call_args in mock_extract.call_args_list:
                extracted_path_str = str(call_args[0][0])
                assert "medqa_usmle" not in extracted_path_str
                assert "l3cube_code_mixed" not in extracted_path_str


import pytest


def test_snapshot_download_nested_files_support(tmp_path):
    """BUG 1 Regression Test: Verify snapshot_download does not filter out nested subdirectories."""
    from download_hf import has_valid_data_files, snapshot_download_fallback

    mock_dest = tmp_path / "test_repo"
    mock_dest.mkdir()

    with patch("huggingface_hub.snapshot_download") as mock_sd:
        mock_sd.return_value = None
        res = snapshot_download_fallback("test_owner/test_repo", mock_dest)
        assert res is True

        # Verify call arguments do NOT restrict allow_patterns (which would break nested directories)
        kwargs = mock_sd.call_args.kwargs
        assert "allow_patterns" not in kwargs

    # Verify has_valid_data_files accepts nested files like data/train.parquet
    nested_file = mock_dest / "data" / "train.parquet"
    nested_file.parent.mkdir(parents=True, exist_ok=True)
    nested_file.write_bytes(b"x" * 200)

    assert has_valid_data_files(mock_dest) is True


def test_bootstrap_raises_error_if_active_datasets_unready(tmp_path):
    """BUG 2 Regression Test: Verify bootstrap_and_audit raises RuntimeError if an active dataset is NOT_READY."""
    from bootstrap import bootstrap_and_audit

    with patch("bootstrap.extract_all_archives"), \
         patch("download_hf.acquire_single_dataset"), \
         patch("bootstrap.RAW", tmp_path), \
         patch("bootstrap.META", tmp_path / "meta"):

        (tmp_path / "meta").mkdir(parents=True, exist_ok=True)

        with patch.dict("bootstrap.ADAPTER_REGISTRY"):
            with pytest.raises(RuntimeError) as exc_info:
                bootstrap_and_audit()

            err_msg = str(exc_info.value)
            assert "Bootstrap audit failed" in err_msg
            assert "NOT_READY" in err_msg
            assert "chatdoctor_healthcaremagic" in err_msg


def test_bootstrap_succeeds_when_all_active_datasets_ready(tmp_path):
    """BUG 2 Regression Test: Verify bootstrap_and_audit returns cleanly when all 10 active datasets are READY."""
    import pandas as pd
    from bootstrap import bootstrap_and_audit

    with patch("bootstrap.extract_all_archives"), \
         patch("download_hf.acquire_single_dataset"), \
         patch("bootstrap.RAW", tmp_path), \
         patch("bootstrap.META", tmp_path / "meta"), \
         patch("bootstrap.get_expected_file") as mock_get_expected:

        (tmp_path / "meta").mkdir(parents=True, exist_ok=True)

        exp_file = tmp_path / "dummy.csv"
        exp_file.write_bytes(b"x" * 200)
        mock_get_expected.return_value = exp_file

        mock_adapter_cls = MagicMock()
        mock_adapter_inst = MagicMock()
        mock_adapter_inst.ingest.return_value = [pd.DataFrame({"a": [1, 2]})]
        mock_adapter_cls.return_value = mock_adapter_inst

        config = Config.from_yaml(PROJECT_ROOT / "config" / "dataset_config.yaml")
        mock_registry = {name: mock_adapter_cls for name in config.active_datasets}

        with patch.dict("bootstrap.ADAPTER_REGISTRY", mock_registry, clear=True):
            bootstrap_and_audit()


def test_meddialog_acquisition_source_is_canonical():
    """Test A: Verify authoritative MedDialog source is wangrongsheng/MedDialog-1.1M."""
    from download_hf import DATASET_SPECS

    meddialog_spec = None
    for spec in DATASET_SPECS:
        if spec[0] == "meddialog_en":
            meddialog_spec = spec
            break

    assert meddialog_spec is not None, "meddialog_en not found in DATASET_SPECS"

    primary_repo = meddialog_spec[1]
    assert primary_repo == "wangrongsheng/MedDialog-1.1M", (
        f"MedDialog primary source is '{primary_repo}', "
        f"expected 'wangrongsheng/MedDialog-1.1M'"
    )

    # petkopetkov/MedDialog should only be a fallback, never the primary
    assert primary_repo != "petkopetkov/MedDialog", (
        "petkopetkov/MedDialog must NOT be the primary MedDialog source"
    )


def test_meddialog_expected_artifact_filename():
    """Test B: Verify expected MedDialog artifact is merged-MedDialog.json."""
    from download_hf import MEDDIALOG_EXPECTED_FILENAME, MEDDIALOG_EXPECTED_RECORDS

    assert MEDDIALOG_EXPECTED_FILENAME == "merged-MedDialog.json"
    assert MEDDIALOG_EXPECTED_RECORDS == 2_725_992


def test_meddialog_tiny_source_rejected(tmp_path):
    """Test D: Verify a tiny substitute source cannot silently satisfy the MedDialog acquisition contract."""
    from download_hf import validate_meddialog_acquisition

    # Simulate the petkopetkov/MedDialog parquet-only acquisition (603 records, <1MB)
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    import pandas as pd
    tiny_df = pd.DataFrame({
        "description": [f"desc_{i}" for i in range(603)],
        "utterances": [[f"utt_{i}"] for i in range(603)],
    })
    tiny_df.to_parquet(data_dir / "train.parquet", index=False)

    # No merged-MedDialog.json present at all
    with pytest.raises(RuntimeError, match="expected artifact"):
        validate_meddialog_acquisition(tmp_path)


def test_meddialog_lfs_pointer_rejected(tmp_path):
    """Test E (acquisition): Verify Git LFS pointer is detected and rejected by acquisition validation."""
    from download_hf import validate_meddialog_acquisition, MEDDIALOG_EXPECTED_FILENAME

    lfs_pointer = (
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:abc123def456789\n"
        "size 4032795837\n"
    )
    json_path = tmp_path / MEDDIALOG_EXPECTED_FILENAME
    json_path.write_text(lfs_pointer, encoding="utf-8")

    with pytest.raises(RuntimeError, match="Git LFS pointer"):
        validate_meddialog_acquisition(tmp_path)


def test_meddialog_undersized_file_rejected(tmp_path):
    """Verify a small merged-MedDialog.json (wrong dataset) fails validation."""
    import json as _json
    from download_hf import validate_meddialog_acquisition, MEDDIALOG_EXPECTED_FILENAME

    # Create a valid JSON array, but far too small
    small_records = [{"instruction": f"q{i}", "input": "", "output": f"a{i}"} for i in range(100)]
    json_path = tmp_path / MEDDIALOG_EXPECTED_FILENAME
    with open(json_path, "w", encoding="utf-8") as f:
        _json.dump(small_records, f)

    with pytest.raises(RuntimeError, match="only.*bytes"):
        validate_meddialog_acquisition(tmp_path)
