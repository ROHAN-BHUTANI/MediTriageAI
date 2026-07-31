"""Unit tests for Clinical Phenotype Augmentation Engine.

Covers:
  - PhenotypeConfig management
  - PhenotypeLibrary knowledge base across 8 specialties
  - ClinicalRuleEngine (contradictory pair prevention, organ-system compatibility, triage rules)
  - PhenotypeQualityValidator (core phenotype retention, rule verification)
  - ClinicalPhenotypeAugmentationEngine (phenotype variant generation, canonical schema compliance)
  - All 4 JSON reports (phenotype_generation, phenotype_statistics, phenotype_distribution, clinical_consistency)
  - MultilingualTranslator pipeline integration
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from meditriage.multilingual.config import MultilingualConfig
from meditriage.multilingual.phenotype.clinical_rules import ClinicalRuleEngine
from meditriage.multilingual.phenotype.phenotype_config import PhenotypeConfig
from meditriage.multilingual.phenotype.phenotype_engine import ClinicalPhenotypeAugmentationEngine
from meditriage.multilingual.phenotype.phenotype_library import PhenotypeDefinition, PhenotypeLibrary
from meditriage.multilingual.phenotype.phenotype_report import generate_phenotype_reports
from meditriage.multilingual.phenotype.phenotype_validator import (
    PhenotypeQualityValidator,
    PhenotypeValidationResult,
)
from meditriage.multilingual.translator import MultilingualTranslator


# ─── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def cardio_row() -> dict:
    return {
        "id": "cardio_001",
        "split": "train",
        "dataset_source": "mtsamples",
        "language": "en",
        "raw_text": "Patient presents with chest pain radiating to left arm and diaphoresis.",
        "department": "CARDIO_PULM",
        "triage_level": "S2",
    }


@pytest.fixture
def peds_row() -> dict:
    return {
        "id": "peds_001",
        "split": "train",
        "dataset_source": "nhamcs_ed",
        "language": "en",
        "raw_text": "Child has high fever of 103F with cough and chills.",
        "department": "PEDS",
        "triage_level": "S3",
    }


@pytest.fixture
def sample_df(cardio_row, peds_row) -> pd.DataFrame:
    return pd.DataFrame([cardio_row, peds_row])


@pytest.fixture
def pheno_cfg(tmp_path: Path) -> PhenotypeConfig:
    return PhenotypeConfig(
        variants_per_sample=2,
        strict_consistency_checking=True,
        output_dir=str(tmp_path / "pheno_reports"),
    )


# ─── Config Tests ──────────────────────────────────────────────────────────

class TestPhenotypeConfig:
    def test_default_config(self):
        cfg = PhenotypeConfig()
        assert len(cfg.enabled_specialties) == 8
        assert "Cardiology" in cfg.enabled_specialties
        assert "Pediatrics" in cfg.enabled_specialties

    def test_save_and_load(self, tmp_path: Path):
        cfg = PhenotypeConfig(variants_per_sample=5)
        path = tmp_path / "cfg.json"
        cfg.save(path)
        loaded = PhenotypeConfig.load(path)
        assert loaded.variants_per_sample == 5


# ─── Phenotype Library Tests ────────────────────────────────────────────────

class TestPhenotypeLibrary:
    def test_library_initialization(self):
        lib = PhenotypeLibrary()
        phenos = lib.get_all_phenotypes()
        assert len(phenos) >= 8

        specs = set(p.specialty for p in phenos)
        assert "Cardiology" in specs
        assert "Neurology" in specs
        assert "Pediatrics" in specs

    def test_match_phenotype(self):
        lib = PhenotypeLibrary()
        match = lib.match_phenotype("Severe chest pain radiating to left shoulder", "CARDIO_PULM")
        assert match is not None
        assert match.specialty == "Cardiology"
        assert "chest pain" in match.core_symptoms


# ─── Clinical Rule Engine Tests ──────────────────────────────────────────────

class TestClinicalRuleEngine:
    def test_contradictory_pair_prevention(self):
        rule_engine = ClinicalRuleEngine()
        lib = PhenotypeLibrary()
        p = lib.get_phenotype_by_id("CARD_ACS")

        # Contradictory: bilateral facial paralysis + appendicitis
        passed, reason = rule_engine.validate_clinical_rules(
            "Patient has bilateral facial paralysis and appendicitis", p
        )
        assert passed is False
        assert "paralysis" in reason or "Contradictory" in reason

    def test_contraindicated_symptom_checking(self):
        rule_engine = ClinicalRuleEngine()
        lib = PhenotypeLibrary()
        p = lib.get_phenotype_by_id("CARD_ACS")

        passed, reason = rule_engine.validate_clinical_rules(
            "Patient has chest pain and petechial rash", p
        )
        assert passed is False
        assert "Contraindicated" in reason

    def test_valid_clinical_presentation(self):
        rule_engine = ClinicalRuleEngine()
        lib = PhenotypeLibrary()
        p = lib.get_phenotype_by_id("CARD_ACS")

        passed, reason = rule_engine.validate_clinical_rules(
            "Substernal chest pressure with diaphoresis and left arm pain", p, department="CARDIO_PULM", triage_level="S2"
        )
        assert passed is True


# ─── Quality Validator Tests ────────────────────────────────────────────────

class TestPhenotypeQualityValidator:
    def test_valid_phenotype_variant(self, cardio_row):
        validator = PhenotypeQualityValidator()
        lib = PhenotypeLibrary()
        p = lib.match_phenotype(cardio_row["raw_text"], cardio_row["department"])

        res = validator.validate_phenotype_variant(
            source_text=cardio_row["raw_text"],
            variant_text="Pressure-like substernal discomfort with nausea.",
            phenotype=p,
            department=cardio_row["department"],
            triage_level=cardio_row["triage_level"],
        )
        assert res.passed is True

    def test_short_variant_rejection(self, cardio_row):
        validator = PhenotypeQualityValidator()
        lib = PhenotypeLibrary()
        p = lib.match_phenotype(cardio_row["raw_text"], cardio_row["department"])

        res = validator.validate_phenotype_variant(
            source_text=cardio_row["raw_text"],
            variant_text="Chest",
            phenotype=p,
        )
        assert res.passed is False
        assert "short" in res.reason


# ─── Engine & Report Tests ──────────────────────────────────────────────────

class TestClinicalPhenotypeAugmentationEngine:
    def test_engine_expansion_schema_and_canonical_columns(self, sample_df, pheno_cfg):
        engine = ClinicalPhenotypeAugmentationEngine(pheno_cfg)
        out_df = engine.expand_dataframe(sample_df, preserve_original=True)

        assert len(out_df) > len(sample_df)
        assert list(out_df.columns) == [
            "id", "split", "dataset_source", "language", "raw_text", "department", "triage_level"
        ]

    def test_ground_truth_preservation(self, sample_df, pheno_cfg):
        engine = ClinicalPhenotypeAugmentationEngine(pheno_cfg)
        out_df = engine.expand_dataframe(sample_df, preserve_original=True)

        # Department and triage level must be identical across all generated variants
        cardio_rows = out_df[out_df["id"].str.startswith("cardio_001")]
        assert len(cardio_rows) > 1
        assert all(cardio_rows["department"] == "CARDIO_PULM")
        assert all(cardio_rows["triage_level"] == "S2")

    def test_reports_generated(self, sample_df, pheno_cfg):
        engine = ClinicalPhenotypeAugmentationEngine(pheno_cfg)
        out_df = engine.expand_dataframe(sample_df, preserve_original=True)

        out_dir = Path(pheno_cfg.output_dir)
        assert (out_dir / "phenotype_generation_report.json").exists()
        assert (out_dir / "phenotype_statistics.json").exists()
        assert (out_dir / "phenotype_distribution.json").exists()
        assert (out_dir / "clinical_consistency_report.json").exists()


# ─── Pipeline Integration Tests ─────────────────────────────────────────────

class TestPipelineIntegration:
    def test_full_multilingual_variation_phenotype_pipeline(self, sample_df, tmp_path: Path):
        m_cfg = MultilingualConfig(
            target_languages=["en", "hi"],
            provider="offline",
            enable_variations=True,
            variation_config={"max_variants_per_sample": 1, "output_dir": str(tmp_path / "var_reports")},
            enable_phenotype_augmentation=True,
            phenotype_config={"variants_per_sample": 1, "output_dir": str(tmp_path / "pheno_reports")},
            cache_dir=str(tmp_path / "cache"),
            output_dir=str(tmp_path / "out"),
        )

        translator = MultilingualTranslator(m_cfg)
        out_df = translator.expand_dataframe(sample_df)

        assert len(out_df) > len(sample_df)
        assert list(out_df.columns) == [
            "id", "split", "dataset_source", "language", "raw_text", "department", "triage_level"
        ]
        for col in out_df.columns:
            assert not out_df[col].isnull().any()
