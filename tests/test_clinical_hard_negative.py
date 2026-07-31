"""Unit tests for Clinical Hard Negative Generation Engine.

Covers:
  - HardNegativeConfig management
  - DifferentialDiagnosisLibrary knowledge base
  - HardNegativeValidator (differential distinction, red flag checking)
  - ClinicalHardNegativeEngine (hard negative generation, canonical schema compliance)
  - Report generation (hard_negative_report, confusion_pair_statistics, differential_coverage)
  - Full pipeline integration with MultilingualTranslator
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from meditriage.multilingual.config import MultilingualConfig
from meditriage.multilingual.hard_negative.hard_negative_config import (
    HardNegativeConfig,
)
from meditriage.multilingual.hard_negative.hard_negative_engine import (
    ClinicalHardNegativeEngine,
)
from meditriage.multilingual.hard_negative.hard_negative_library import (
    DifferentialDiagnosisLibrary,
)
from meditriage.multilingual.hard_negative.hard_negative_validator import (
    HardNegativeValidator,
)
from meditriage.multilingual.translator import MultilingualTranslator

# ─── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def ami_row() -> dict:
    return {
        "id": "ami_001",
        "split": "train",
        "dataset_source": "mtsamples",
        "language": "en",
        "raw_text": "Patient presents with acute substernal chest pain radiating to left arm with diaphoresis.",
        "department": "CARDIO_PULM",
        "triage_level": "S2",
    }


@pytest.fixture
def stroke_row() -> dict:
    return {
        "id": "stroke_001",
        "split": "train",
        "dataset_source": "nhamcs_ed",
        "language": "en",
        "raw_text": "Sudden onset right arm weakness and slurred speech starting 1 hour ago.",
        "department": "NEURO",
        "triage_level": "S1",
    }


@pytest.fixture
def sample_df(ami_row, stroke_row) -> pd.DataFrame:
    return pd.DataFrame([ami_row, stroke_row])


@pytest.fixture
def hn_cfg(tmp_path: Path) -> HardNegativeConfig:
    return HardNegativeConfig(
        negatives_per_sample=2,
        strict_validation=True,
        output_dir=str(tmp_path / "hn_reports"),
    )


# ─── Config Tests ──────────────────────────────────────────────────────────


class TestHardNegativeConfig:
    def test_default_config(self):
        cfg = HardNegativeConfig()
        assert cfg.negatives_per_sample == 2
        assert cfg.strict_validation is True

    def test_save_and_load(self, tmp_path: Path):
        cfg = HardNegativeConfig(negatives_per_sample=4)
        path = tmp_path / "hn_cfg.json"
        cfg.save(path)
        loaded = HardNegativeConfig.load(path)
        assert loaded.negatives_per_sample == 4


# ─── Library Tests ─────────────────────────────────────────────────────────


class TestDifferentialDiagnosisLibrary:
    def test_library_initialization(self):
        lib = DifferentialDiagnosisLibrary()
        diffs = lib.get_differentials_for_text(
            "Severe chest pain radiating to arm", "CARDIO_PULM"
        )
        assert len(diffs) >= 2
        diff_names = [d.name for d in diffs]
        assert any("GERD" in n or "Reflux" in n for n in diff_names)

    def test_stroke_differentials(self):
        lib = DifferentialDiagnosisLibrary()
        diffs = lib.get_differentials_for_text(
            "Sudden right arm weakness and facial droop", "NEURO"
        )
        assert len(diffs) >= 2
        diff_names = [d.name for d in diffs]
        assert any("Bell's Palsy" in n or "Hypoglycemia" in n for n in diff_names)


# ─── Validator Tests ───────────────────────────────────────────────────────


class TestHardNegativeValidator:
    def test_valid_hard_negative(self, ami_row):
        validator = HardNegativeValidator()
        lib = DifferentialDiagnosisLibrary()
        diffs = lib.get_differentials_for_text(
            ami_row["raw_text"], ami_row["department"]
        )
        gerd_diff = next(d for d in diffs if "GERD" in d.name or "Reflux" in d.name)

        res = validator.validate_hard_negative(
            source_text=ami_row["raw_text"],
            negative_text="Substernal chest burning after heavy meal, relieved by antacid liquid.",
            diff_entry=gerd_diff,
            original_department=ami_row["department"],
        )
        assert res.passed is True

    def test_red_flag_rejection(self, ami_row):
        validator = HardNegativeValidator()
        lib = DifferentialDiagnosisLibrary()
        diffs = lib.get_differentials_for_text(
            ami_row["raw_text"], ami_row["department"]
        )
        gerd_diff = next(d for d in diffs if "GERD" in d.name or "Reflux" in d.name)

        # Reject if negative contains 'radiation to arm' which is forbidden for GERD differential
        res = validator.validate_hard_negative(
            source_text=ami_row["raw_text"],
            negative_text="Substernal burning with radiation to arm.",
            diff_entry=gerd_diff,
        )
        assert res.passed is False
        assert "red-flag" in res.reason.lower()


# ─── Engine & Report Tests ──────────────────────────────────────────────────


class TestClinicalHardNegativeEngine:
    def test_engine_expansion_and_schema(self, sample_df, hn_cfg):
        engine = ClinicalHardNegativeEngine(hn_cfg)
        out_df = engine.expand_dataframe(sample_df, preserve_original=True)

        assert len(out_df) > len(sample_df)
        assert list(out_df.columns) == [
            "id",
            "split",
            "dataset_source",
            "language",
            "raw_text",
            "department",
            "triage_level",
        ]

    def test_hard_negative_department_reassignment(self, sample_df, hn_cfg):
        engine = ClinicalHardNegativeEngine(hn_cfg)
        out_df = engine.expand_dataframe(sample_df, preserve_original=True)

        # Differential negatives should be correctly assigned to their target department
        hn_rows = out_df[out_df["id"].str.contains("hardneg")]
        assert len(hn_rows) >= 2
        depts = set(hn_rows["department"].tolist())
        # Differential negatives for AMI include GERD (GENERAL) or Costochondritis (ORTHO)
        assert "GENERAL" in depts or "ORTHO" in depts or "NEURO" in depts

    def test_reports_generated(self, sample_df, hn_cfg):
        engine = ClinicalHardNegativeEngine(hn_cfg)
        engine.expand_dataframe(sample_df, preserve_original=True)

        out_dir = Path(hn_cfg.output_dir)
        assert (out_dir / "hard_negative_report.json").exists()
        assert (out_dir / "confusion_pair_statistics.json").exists()
        assert (out_dir / "differential_coverage.json").exists()


# ─── Full Pipeline Integration Tests ────────────────────────────────────────


class TestFullPipelineIntegration:
    def test_complete_4_stage_pipeline(self, sample_df, tmp_path: Path):
        m_cfg = MultilingualConfig(
            target_languages=["en", "hi"],
            provider="offline",
            enable_variations=True,
            variation_config={
                "max_variants_per_sample": 1,
                "output_dir": str(tmp_path / "var"),
            },
            enable_phenotype_augmentation=True,
            phenotype_config={
                "variants_per_sample": 1,
                "output_dir": str(tmp_path / "pheno"),
            },
            enable_hard_negatives=True,
            hard_negative_config={
                "negatives_per_sample": 1,
                "output_dir": str(tmp_path / "hn"),
            },
            cache_dir=str(tmp_path / "cache"),
            output_dir=str(tmp_path / "out"),
        )

        translator = MultilingualTranslator(m_cfg)
        out_df = translator.expand_dataframe(sample_df)

        assert len(out_df) > len(sample_df)
        assert list(out_df.columns) == [
            "id",
            "split",
            "dataset_source",
            "language",
            "raw_text",
            "department",
            "triage_level",
        ]
        for col in out_df.columns:
            assert not out_df[col].isnull().any()
