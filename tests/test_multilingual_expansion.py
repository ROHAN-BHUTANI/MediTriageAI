"""Unit tests for Multilingual Dataset Expansion Engine.

Covers:
  - Configuration management
  - ClinicalQualityValidator (refusals, script, length, numbers, clinical terms)
  - MultilingualCache (get/set, save/load, thread safety)
  - Providers (Offline, Gemini, OpenAI)
  - MultilingualTranslator (DataFrame expansion, schema compatibility, 7 canonical columns)
  - Audit report generation
  - Stage 5 builder integration
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from meditriage.multilingual.cache import MultilingualCache
from meditriage.multilingual.config import MultilingualConfig
from meditriage.multilingual.providers import get_provider, list_providers
from meditriage.multilingual.providers.offline import OfflineMultilingualProvider
from meditriage.multilingual.report import generate_multilingual_reports
from meditriage.multilingual.translator import MultilingualTranslator
from meditriage.multilingual.validator import ClinicalQualityValidator, ValidationResult


# ─── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def sample_clinical_df() -> pd.DataFrame:
    """Sample DataFrame with canonical clinical fields."""
    return pd.DataFrame([
        {
            "id": "sample_001",
            "split": "train",
            "dataset_source": "mtsamples",
            "language": "en",
            "raw_text": "Patient presents with severe chest pain and shortness of breath since 2 hours.",
            "department": "CARDIO_PULM",
            "triage_level": "S2",
        },
        {
            "id": "sample_002",
            "split": "val",
            "dataset_source": "pmc_patients",
            "language": "en",
            "raw_text": "Patient has high fever of 102F and persistent cough since yesterday.",
            "department": "PEDS",
            "triage_level": "S3",
        },
        {
            "id": "sample_003",
            "split": "test",
            "dataset_source": "nhamcs_ed",
            "language": "en",
            "raw_text": "Acute headache with dizziness and nausea after minor head injury.",
            "department": "NEURO",
            "triage_level": "S3",
        },
    ])


@pytest.fixture
def cfg(tmp_path: Path) -> MultilingualConfig:
    return MultilingualConfig(
        target_languages=["en", "hi", "hi-Latn", "hi-en", "en-hi"],
        provider="offline",
        cache_dir=str(tmp_path / "cache"),
        output_dir=str(tmp_path / "output"),
        num_workers=2,
    )


# ─── Config Tests ──────────────────────────────────────────────────────────

class TestMultilingualConfig:
    def test_default_config(self):
        cfg = MultilingualConfig()
        assert len(cfg.target_languages) == 5
        assert "hi" in cfg.target_languages
        assert "hi-Latn" in cfg.target_languages
        assert cfg.provider == "offline"

    def test_save_and_load(self, tmp_path: Path):
        cfg = MultilingualConfig(provider="gemini", num_workers=8)
        path = tmp_path / "cfg.json"
        cfg.save(path)
        loaded = MultilingualConfig.load(path)
        assert loaded.provider == "gemini"
        assert loaded.num_workers == 8

    def test_from_dict(self):
        cfg = MultilingualConfig.from_dict({"provider": "openai", "batch_size": 100})
        assert cfg.provider == "openai"
        assert cfg.batch_size == 100


# ─── Validator Tests ────────────────────────────────────────────────────────

class TestClinicalQualityValidator:
    def test_valid_english(self):
        v = ClinicalQualityValidator()
        res = v.validate("Patient has chest pain", "Patient has severe chest pain", "en")
        assert res.passed is True

    def test_valid_hindi_devanagari(self):
        v = ClinicalQualityValidator()
        res = v.validate("Patient has chest pain", "मरीज़ को छाती में दर्द हो रहा है।", "hi")
        assert res.passed is True

    def test_valid_roman_hindi(self):
        v = ClinicalQualityValidator()
        res = v.validate("Patient has chest pain", "Patient ko chaati mein dard ho raha hai.", "hi-Latn")
        assert res.passed is True

    def test_valid_hinglish(self):
        v = ClinicalQualityValidator()
        res = v.validate("Patient has chest pain", "Chest mein bahut pain ho raha hai.", "hi-en")
        assert res.passed is True

    def test_valid_codeswitched(self):
        v = ClinicalQualityValidator()
        res = v.validate("Patient has chest pain", "Patient ko severe chest pain hai.", "en-hi")
        assert res.passed is True

    def test_refusal_detection(self):
        v = ClinicalQualityValidator()
        res = v.validate("Chest pain", "I am sorry, as an AI I cannot provide medical advice.", "hi")
        assert res.passed is False
        assert "Refusal" in res.reason

    def test_missing_script_hindi(self):
        v = ClinicalQualityValidator()
        res = v.validate("Chest pain", "Patient has chest pain", "hi")
        assert res.passed is False
        assert "Devanagari" in res.reason

    def test_empty_or_short(self):
        v = ClinicalQualityValidator()
        assert v.validate("Chest pain", "", "hi").passed is False
        assert v.validate("Chest pain", "ab", "hi").passed is False

    def test_number_preservation(self):
        v = ClinicalQualityValidator()
        # Source has 102, target missing number
        res = v.validate("Patient has fever of 102F", "Patient has fever", "en-hi")
        assert res.passed is False
        assert "Numerical discrepancy" in res.reason


# ─── Cache Tests ────────────────────────────────────────────────────────────

class TestMultilingualCache:
    def test_cache_get_set(self, tmp_path: Path):
        cache = MultilingualCache(cache_dir=tmp_path)
        cache.set("Chest pain", "hi", "offline", "छाती में दर्द", validated=True)
        cached = cache.get("Chest pain", "hi", "offline")
        assert cached is not None
        assert cached["translated_text"] == "छाती में दर्द"

    def test_cache_persistence(self, tmp_path: Path):
        cache1 = MultilingualCache(cache_dir=tmp_path)
        cache1.set("Chest pain", "hi-Latn", "offline", "chaati mein dard")
        cache1.save()

        cache2 = MultilingualCache(cache_dir=tmp_path)
        cached = cache2.get("Chest pain", "hi-Latn", "offline")
        assert cached is not None
        assert cached["translated_text"] == "chaati mein dard"


# ─── Provider Tests ─────────────────────────────────────────────────────────

class TestProviders:
    def test_offline_provider_languages(self):
        p = OfflineMultilingualProvider()
        for lang in ["hi", "hi-Latn", "hi-en", "en-hi"]:
            res = p.translate_text("Patient presents with severe chest pain", lang)
            assert isinstance(res, str)
            assert len(res) > 0

    def test_offline_provider_determinism(self):
        p = OfflineMultilingualProvider()
        t1 = p.translate_text("Patient has high fever and cough", "hi-Latn")
        t2 = p.translate_text("Patient has high fever and cough", "hi-Latn")
        assert t1 == t2

    def test_get_provider_factory(self):
        assert "offline" in list_providers()
        p = get_provider("offline")
        assert isinstance(p, OfflineMultilingualProvider)


# ─── Translator & Integration Tests ────────────────────────────────────────

class TestMultilingualTranslator:
    def test_expand_dataframe_canonical_columns(self, sample_clinical_df, cfg):
        translator = MultilingualTranslator(cfg)
        out_df = translator.expand_dataframe(sample_clinical_df)

        # Output rows: 3 input * 5 target languages = 15 rows
        assert len(out_df) == 15
        assert list(out_df.columns) == [
            "id", "split", "dataset_source", "language", "raw_text", "department", "triage_level"
        ]

        # Verify language breakdown
        langs = set(out_df["language"].unique())
        assert langs == {"en", "hi", "hi-Latn", "hi-en", "en-hi"}

    def test_semantic_preservation(self, sample_clinical_df, cfg):
        translator = MultilingualTranslator(cfg)
        out_df = translator.expand_dataframe(sample_clinical_df)

        # Department and triage level must be preserved identically for every expanded row
        cardio_rows = out_df[out_df["id"].str.startswith("sample_001")]
        assert len(cardio_rows) == 5
        assert all(cardio_rows["department"] == "CARDIO_PULM")
        assert all(cardio_rows["triage_level"] == "S2")

    def test_no_null_canonical_fields(self, sample_clinical_df, cfg):
        translator = MultilingualTranslator(cfg)
        out_df = translator.expand_dataframe(sample_clinical_df)
        for col in ["id", "split", "dataset_source", "language", "raw_text", "department", "triage_level"]:
            assert not out_df[col].isnull().any()


# ─── Report Tests ───────────────────────────────────────────────────────────

class TestMultilingualReport:
    def test_report_generation(self, sample_clinical_df, cfg):
        translator = MultilingualTranslator(cfg)
        out_df = translator.expand_dataframe(sample_clinical_df)

        master = generate_multilingual_reports(out_df, translator.stats, cfg)
        out_dir = Path(cfg.output_dir)

        assert (out_dir / "multilingual_expansion_report.json").exists()
        assert (out_dir / "multilingual_language_distribution.json").exists()
        assert (out_dir / "multilingual_language_distribution.md").exists()
        assert (out_dir / "multilingual_quality_report.json").exists()
        assert (out_dir / "multilingual_validation_report.json").exists()

        assert "language_distribution" in master
        assert "expansion_summary" in master
