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
