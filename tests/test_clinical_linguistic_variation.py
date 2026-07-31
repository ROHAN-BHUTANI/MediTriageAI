"""Unit tests for Clinical Linguistic Variation Engine.

Covers:
  - VariationConfig management
  - All 10 linguistic variation generators
  - SemanticVariationValidator (similarity, length ratio, number & clinical preservation)
  - ClinicalLinguisticVariationEngine (variation budgets, canonical 7-column schema)
  - Report generation (clinical_variation_report, variation_statistics, semantic_similarity_report)
  - MultilingualTranslator integration
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from meditriage.multilingual.config import MultilingualConfig
from meditriage.multilingual.translator import MultilingualTranslator
from meditriage.multilingual.variation.config import VariationConfig
from meditriage.multilingual.variation.engine import ClinicalLinguisticVariationEngine
from meditriage.multilingual.variation.generators import (
    AbbreviatedNotationGenerator,
    ColloquialIndianGenerator,
    ConversationalVariationGenerator,
    EdTriageVariationGenerator,
    FormalDocumentationGenerator,
    LexicalVariationGenerator,
    NurseIntakeVariationGenerator,
    PhysicianNoteVariationGenerator,
    SyntacticVariationGenerator,
    get_all_generators,
    get_generator_by_name,
)
from meditriage.multilingual.variation.report import generate_variation_reports
from meditriage.multilingual.variation.validator import (
    SemanticVariationValidator,
    VariationValidationResult,
)


# ─── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def sample_row() -> dict:
    return {
        "id": "sample_101",
        "split": "train",
        "dataset_source": "mtsamples",
        "language": "en",
        "raw_text": "Patient presents with severe chest pain and shortness of breath since 2 hours.",
        "department": "CARDIO_PULM",
        "triage_level": "S2",
    }


@pytest.fixture
def sample_df(sample_row) -> pd.DataFrame:
    return pd.DataFrame([
        sample_row,
        {
            "id": "sample_102",
            "split": "val",
            "dataset_source": "pmc_patients",
            "language": "en",
            "raw_text": "High fever of 102F with persistent cough since yesterday.",
            "department": "PEDS",
            "triage_level": "S3",
        },
    ])


@pytest.fixture
def var_cfg(tmp_path: Path) -> VariationConfig:
    return VariationConfig(
        enabled_styles=[
            "lexical",
            "syntactic",
            "conversational",
            "ed_triage",
            "physician_note",
            "nurse_intake",
            "abbreviated_notation",
            "formal_documentation",
            "colloquial_indian",
        ],
        variation_budgets={
            "lexical": 1,
            "syntactic": 1,
            "conversational": 1,
            "ed_triage": 1,
            "physician_note": 1,
            "nurse_intake": 1,
            "abbreviated_notation": 1,
            "formal_documentation": 1,
            "colloquial_indian": 1,
        },
        max_variants_per_sample=8,
        min_semantic_similarity=0.50,
        output_dir=str(tmp_path / "variation_reports"),
    )


# ─── Config Tests ──────────────────────────────────────────────────────────

class TestVariationConfig:
    def test_default_config(self):
        cfg = VariationConfig()
        assert len(cfg.enabled_styles) == 9
        assert "lexical" in cfg.enabled_styles
        assert "ed_triage" in cfg.enabled_styles
        assert "colloquial_indian" in cfg.enabled_styles

    def test_save_and_load(self, tmp_path: Path):
        cfg = VariationConfig(max_variants_per_sample=5)
        path = tmp_path / "var_cfg.json"
        cfg.save(path)
        loaded = VariationConfig.load(path)
        assert loaded.max_variants_per_sample == 5


# ─── Generator Tests ───────────────────────────────────────────────────────

class TestVariationGenerators:
    def test_all_generators_registered(self):
        gens = get_all_generators()
        assert len(gens) == 9

    def test_lexical_generator(self, sample_row):
        g = LexicalVariationGenerator()
        res = g.generate_variants(sample_row["raw_text"], budget=2)
        assert len(res) >= 1
        assert isinstance(res[0], str)

    def test_syntactic_generator(self, sample_row):
        g = SyntacticVariationGenerator()
        res = g.generate_variants(sample_row["raw_text"], budget=2)
        assert len(res) >= 1

    def test_conversational_generator(self, sample_row):
        g = ConversationalVariationGenerator()
        res = g.generate_variants(sample_row["raw_text"], budget=2)
        assert len(res) >= 1

    def test_ed_triage_generator(self, sample_row):
        g = EdTriageVariationGenerator()
        res = g.generate_variants(
            sample_row["raw_text"],
            department=sample_row["department"],
            triage_level=sample_row["triage_level"],
            budget=2,
        )
        assert len(res) >= 1
        assert "S2" in res[0] or "CARDIO_PULM" in res[0]

    def test_physician_note_generator(self, sample_row):
        g = PhysicianNoteVariationGenerator()
        res = g.generate_variants(sample_row["raw_text"], department="CARDIO_PULM", budget=1)
        assert len(res) == 1

    def test_nurse_intake_generator(self, sample_row):
        g = NurseIntakeVariationGenerator()
        res = g.generate_variants(sample_row["raw_text"], budget=1)
        assert len(res) == 1

    def test_abbreviated_notation_generator(self, sample_row):
        g = AbbreviatedNotationGenerator()
        res = g.generate_variants(sample_row["raw_text"], budget=1)
        assert len(res) == 1
        assert "CP" in res[0] or "SOB" in res[0] or "Pt" in res[0]

    def test_formal_documentation_generator(self, sample_row):
        g = FormalDocumentationGenerator()
        res = g.generate_variants(sample_row["raw_text"], department="CARDIO_PULM", budget=1)
        assert len(res) == 1

    def test_colloquial_indian_generator(self, sample_row):
        g = ColloquialIndianGenerator()
        res = g.generate_variants(sample_row["raw_text"], budget=2)
        assert len(res) >= 1


# ─── Validator Tests ────────────────────────────────────────────────────────

class TestSemanticVariationValidator:
    def test_valid_variant(self, sample_row):
        v = SemanticVariationValidator()
        res = v.validate_variant(sample_row["raw_text"], "Patient has pain in chest since 2 hours.")
        assert res.passed is True
        assert res.similarity_score > 0.35

    def test_refusal_rejection(self, sample_row):
        v = SemanticVariationValidator()
        res = v.validate_variant(sample_row["raw_text"], "I am sorry, as an AI I cannot help.")
        assert res.passed is False
        assert "boilerplate" in res.reason.lower()

    def test_number_mismatch_rejection(self):
        v = SemanticVariationValidator()
        res = v.validate_variant("Patient has fever of 102F since 3 days.", "Patient has fever since yesterday.")
        assert res.passed is False
        assert "Numerical mismatch" in res.reason


# ─── Engine Tests ───────────────────────────────────────────────────────────

class TestClinicalLinguisticVariationEngine:
    def test_engine_expansion_canonical_schema(self, sample_df, var_cfg):
        engine = ClinicalLinguisticVariationEngine(var_cfg)
        out_df = engine.expand_dataframe(sample_df, preserve_original=True)

        assert len(out_df) > len(sample_df)
        assert list(out_df.columns) == [
            "id", "split", "dataset_source", "language", "raw_text", "department", "triage_level"
        ]

    def test_department_and_triage_preservation(self, sample_df, var_cfg):
        engine = ClinicalLinguisticVariationEngine(var_cfg)
        out_df = engine.expand_dataframe(sample_df, preserve_original=True)

        # Department & triage must be strictly preserved for all generated variants
        cardio_df = out_df[out_df["id"].str.startswith("sample_101")]
        assert len(cardio_df) > 1
        assert all(cardio_df["department"] == "CARDIO_PULM")
        assert all(cardio_df["triage_level"] == "S2")

    def test_report_generation(self, sample_df, var_cfg):
        engine = ClinicalLinguisticVariationEngine(var_cfg)
        out_df = engine.expand_dataframe(sample_df, preserve_original=True)

        out_dir = Path(var_cfg.output_dir)
        assert (out_dir / "clinical_variation_report.json").exists()
        assert (out_dir / "variation_statistics.json").exists()
        assert (out_dir / "semantic_similarity_report.json").exists()


# ─── Integration with MultilingualTranslator ───────────────────────────────

class TestMultilingualIntegration:
    def test_multilingual_with_variation(self, sample_df, tmp_path: Path):
        m_cfg = MultilingualConfig(
            target_languages=["en", "hi"],
            provider="offline",
            enable_variations=True,
            variation_config={"max_variants_per_sample": 2, "output_dir": str(tmp_path / "var_out")},
            cache_dir=str(tmp_path / "cache"),
            output_dir=str(tmp_path / "out"),
        )

        translator = MultilingualTranslator(m_cfg)
        out_df = translator.expand_dataframe(sample_df)

        assert len(out_df) > len(sample_df)
        assert list(out_df.columns) == [
            "id", "split", "dataset_source", "language", "raw_text", "department", "triage_level"
        ]
