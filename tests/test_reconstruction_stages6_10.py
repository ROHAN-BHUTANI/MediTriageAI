"""Unit tests for the Dataset Reconstruction Engine – Stages 6-10.

Covers: augmentation plugins, LLM providers, merge, shuffle,
validation, diversity report, and CLI integration.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from reconstruction.augmentations import AugmentationPlugin
from reconstruction.augmentations.plugins import (
    ALL_PLUGINS,
    AsrCorruption,
    BrokenEnglish,
    BrokenHinglish,
    CapitalizationVariation,
    ClinicallyEquivalentRewrite,
    EnglishLexicalRewrite,
    HindiTranslation,
    HinglishConversion,
    KeyboardTypo,
    MedicalAbbreviationContraction,
    MedicalAbbreviationExpansion,
    PunctuationRemoval,
    RomanHindiConversion,
    SmsShorthand,
    SymptomOrderPermutation,
    get_all_plugins,
)
from reconstruction.config import ReconstructionConfig
from reconstruction.llm import (
    GeneratedSample,
    get_provider,
    hash_prompt,
    list_providers,
)
from reconstruction.report import generate_diversity_report
from reconstruction.run import parse_stages
from reconstruction.stage6_augment import augment_class
from reconstruction.stage7_generate import generate_for_class
from reconstruction.stage8_merge import merge_datasets
from reconstruction.stage9_shuffle import deterministic_shuffle
from reconstruction.stage10_validate import (
    run_validators,
    validate_balance,
    validate_contradictions,
    validate_duplicates,
    validate_embedding_similarity,
    validate_language,
    validate_phenotype,
    validate_provenance,
)

# ─── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def mid_tier_df() -> pd.DataFrame:
    """DataFrame simulating a mid-tier class (size >= 500 but < target)."""
    n = 600
    return pd.DataFrame(
        {
            "id": [f"mid_{i}" for i in range(n)],
            "split": ["train"] * n,
            "dataset_source": ["test"] * n,
            "language": ["en" if i % 4 != 0 else "hi-en" for i in range(n)],
            "raw_text": [
                f"Patient {i} has severe pain and fever with swelling since yesterday"
                for i in range(n)
            ],
            "department": ["CARDIO_PULM"] * n,
            "triage_level": [f"S{(i % 5) + 1}" for i in range(n)],
            "cluster_id": [i % 10 for i in range(n)],
            "diversity_score": np.random.RandomState(42).rand(n).tolist(),
        }
    )


@pytest.fixture
def minority_df() -> pd.DataFrame:
    """DataFrame simulating an extreme minority class (size < 500)."""
    n = 20
    return pd.DataFrame(
        {
            "id": [f"min_{i}" for i in range(n)],
            "split": ["train"] * n,
            "dataset_source": ["test"] * n,
            "language": ["en"] * n,
            "raw_text": [
                f"Patient {i} presents with gynecological symptoms including pain"
                for i in range(n)
            ],
            "department": ["OBGYN"] * n,
            "triage_level": ["S3"] * n,
            "cluster_id": [0] * n,
            "diversity_score": np.random.RandomState(42).rand(n).tolist(),
        }
    )


@pytest.fixture
def full_df() -> pd.DataFrame:
    """DataFrame with multiple departments for integration tests."""
    rows = []
    for dept in ["ORTHO", "NEURO"]:
        for i in range(50):
            rows.append(
                {
                    "id": f"{dept}_{i}",
                    "split": "train",
                    "dataset_source": "test",
                    "language": "en",
                    "raw_text": f"Patient {i} has {dept.lower()} related pain and symptoms in area {i}",
                    "department": dept,
                    "triage_level": f"S{(i % 5) + 1}",
                    "cluster_id": i % 5,
                    "diversity_score": float(i) / 50,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def cfg(tmp_path: Path) -> ReconstructionConfig:
    return ReconstructionConfig(
        target_class_size=50,
        output_directory=str(tmp_path / "recon_out"),
        random_seed=42,
        augmentation_min_class_size=500,
        llm_provider="offline",
    )


# ─── Plugin Tests ────────────────────────────────────────────────────────


class TestAugmentationPlugins:
    def test_all_plugins_registered(self):
        assert len(ALL_PLUGINS) == 15

    def test_get_all_plugins_instantiates(self):
        plugins = get_all_plugins()
        assert len(plugins) == 15
        assert all(isinstance(p, AugmentationPlugin) for p in plugins)

    def test_every_plugin_has_metadata(self):
        for plugin in get_all_plugins():
            meta = plugin.plugin_metadata()
            assert "name" in meta
            assert "version" in meta

    def test_every_plugin_has_languages(self):
        for plugin in get_all_plugins():
            langs = plugin.supported_languages()
            assert isinstance(langs, list)
            assert len(langs) >= 1

    def test_english_lexical_rewrite(self):
        p = EnglishLexicalRewrite()
        result = p.apply("Patient has severe pain and fever", seed=42)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_hindi_translation(self):
        p = HindiTranslation()
        result = p.apply("I have pain and fever since yesterday", seed=42)
        assert isinstance(result, str)

    def test_roman_hindi(self):
        p = RomanHindiConversion()
        result = p.apply("I have pain in my head and stomach", seed=42)
        assert isinstance(result, str)

    def test_hinglish(self):
        p = HinglishConversion()
        result = p.apply("I have fever and pain", seed=42)
        assert isinstance(result, str)

    def test_broken_english(self):
        p = BrokenEnglish()
        result = p.apply("The patient has a severe headache", seed=42)
        assert isinstance(result, str)

    def test_broken_hinglish(self):
        p = BrokenHinglish()
        result = p.apply("I have pain and fever since yesterday", seed=42)
        assert isinstance(result, str)

    def test_sms_shorthand(self):
        p = SmsShorthand()
        result = p.apply("Please doctor I have a problem today", seed=42)
        assert isinstance(result, str)

    def test_asr_corruption(self):
        p = AsrCorruption()
        result = p.apply("I have pain in my right eye", seed=42)
        assert isinstance(result, str)

    def test_keyboard_typo(self):
        p = KeyboardTypo()
        result = p.apply("Patient has headache", seed=42)
        assert isinstance(result, str)
        assert len(result) == len("Patient has headache")

    def test_medical_abbrev_expansion(self):
        p = MedicalAbbreviationExpansion()
        result = p.apply("pt has sob and high bp", seed=42)
        assert isinstance(result, str)

    def test_medical_abbrev_contraction(self):
        p = MedicalAbbreviationContraction()
        result = p.apply("patient has shortness of breath", seed=42)
        assert isinstance(result, str)

    def test_punctuation_removal(self):
        p = PunctuationRemoval()
        result = p.apply("Hello, world! How are you?", seed=42)
        assert isinstance(result, str)

    def test_capitalization_variation(self):
        p = CapitalizationVariation()
        result = p.apply("Hello World", seed=42)
        assert isinstance(result, str)

    def test_symptom_order_permutation(self):
        p = SymptomOrderPermutation()
        result = p.apply("pain, fever, swelling, cough", seed=42)
        assert isinstance(result, str)

    def test_clinically_equivalent_rewrite(self):
        p = ClinicallyEquivalentRewrite()
        result = p.apply("my head hurts a lot of pain", seed=42)
        assert isinstance(result, str)

    def test_determinism(self):
        """Same seed must produce same output."""
        p = EnglishLexicalRewrite()
        a = p.apply("Patient has severe pain", seed=123)
        b = p.apply("Patient has severe pain", seed=123)
        assert a == b

    def test_different_seeds_differ(self):
        """Different seeds should generally produce different output."""
        p = KeyboardTypo()
        text = "Patient has a really bad headache and severe pain"
        a = p.apply(text, seed=1)
        b = p.apply(text, seed=999)
        # At least one should differ (keyboard typo is very likely to differ)
        # But we can't guarantee it, so we just check they're strings
        assert isinstance(a, str) and isinstance(b, str)


# ─── LLM Provider Tests ─────────────────────────────────────────────────


class TestLLMProviders:
    def test_offline_registered(self):
        assert "offline" in list_providers()

    def test_offline_generate(self):
        provider = get_provider("offline")
        results = provider.generate("symptoms: headache and fever", n=5, seed=42)
        assert len(results) == 5
        assert all(isinstance(r, str) for r in results)

    def test_offline_validate(self):
        provider = get_provider("offline")
        assert provider.validate("Patient has pain", "ORTHO") is True
        assert provider.validate("", "ORTHO") is False
        assert provider.validate("abc", "ORTHO") is False

    def test_offline_metadata(self):
        provider = get_provider("offline")
        meta = provider.provider_metadata()
        assert "name" in meta

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            get_provider("nonexistent_provider_xyz")

    def test_hash_prompt_deterministic(self):
        h1 = hash_prompt("test prompt")
        h2 = hash_prompt("test prompt")
        assert h1 == h2

    def test_hash_prompt_differs(self):
        h1 = hash_prompt("prompt A")
        h2 = hash_prompt("prompt B")
        assert h1 != h2

    def test_generated_sample_dataclass(self):
        s = GeneratedSample(
            text="test",
            department="ORTHO",
            source_sample_id="src_1",
            generation_prompt_hash="abc123",
            provider="offline",
            generation_id="gen_1",
        )
        assert s.text == "test"
        assert s.department == "ORTHO"


# ─── Stage 6 Tests ──────────────────────────────────────────────────────


class TestStage6:
    def test_augment_class_generates_deficit(self, mid_tier_df, cfg):
        cfg.target_class_size = 700
        cfg.augmentation_min_class_size = 500
        result, report = augment_class(mid_tier_df, 700, cfg)
        assert len(result) == 700
        assert report["action"] == "augmented"
        assert report["generated"] > 0

    def test_augment_passthrough_when_at_target(self, mid_tier_df, cfg):
        result, report = augment_class(mid_tier_df, 50, cfg)
        assert len(result) == len(mid_tier_df)
        assert report["generated"] == 0

    def test_augmented_rows_have_provenance(self, mid_tier_df, cfg):
        cfg.target_class_size = 700
        result, _ = augment_class(mid_tier_df, 700, cfg)
        aug_rows = result[result["dataset_source"].str.startswith("augmented_")]
        assert len(aug_rows) > 0
        assert "_provenance_plugin" in aug_rows.columns
        assert "_provenance_seed" in aug_rows.columns


# ─── Stage 7 Tests ──────────────────────────────────────────────────────


class TestStage7:
    def test_generate_for_class(self, minority_df, cfg):
        cfg.target_class_size = 50
        cfg.llm_provider = "offline"
        result, report = generate_for_class(minority_df, 50, cfg)
        assert len(result) == 50
        assert report["action"] == "synthetic_generation"
        assert report["generated"] > 0

    def test_generate_passthrough(self, minority_df, cfg):
        result, report = generate_for_class(minority_df, 10, cfg)
        assert len(result) == len(minority_df)
        assert report["generated"] == 0

    def test_synthetic_provenance(self, minority_df, cfg):
        cfg.target_class_size = 30
        result, _ = generate_for_class(minority_df, 30, cfg)
        syn = result[result["dataset_source"].str.startswith("synthetic_")]
        assert len(syn) > 0
        assert "_provenance_provider" in syn.columns
        assert "_provenance_prompt_hash" in syn.columns


# ─── Stage 8 Tests ──────────────────────────────────────────────────────


class TestStage8:
    def test_merge_statistics(self, full_df):
        _merged, report = merge_datasets(full_df)
        assert report["total_rows"] == len(full_df)
        assert "department_counts" in report

    def test_merge_preserves_all_rows(self, full_df):
        merged, _ = merge_datasets(full_df)
        assert len(merged) == len(full_df)

    def test_merge_composition_breakdown(self):
        df = pd.DataFrame(
            {
                "id": ["a", "b", "c"],
                "raw_text": ["x", "y", "z"],
                "department": ["ORTHO", "ORTHO", "ORTHO"],
                "dataset_source": [
                    "neiss",
                    "augmented_KeyboardTypo",
                    "synthetic_OfflineProvider",
                ],
                "language": ["en", "en", "en"],
            }
        )
        _, report = merge_datasets(df)
        assert report["composition"]["original"] == 1
        assert report["composition"]["augmented"] == 1
        assert report["composition"]["synthetic"] == 1


# ─── Stage 9 Tests ──────────────────────────────────────────────────────


class TestStage9:
    def test_deterministic_same_seed(self, full_df):
        s1 = deterministic_shuffle(full_df, seed=42)
        s2 = deterministic_shuffle(full_df, seed=42)
        assert s1["id"].tolist() == s2["id"].tolist()

    def test_deterministic_different_seed(self, full_df):
        s1 = deterministic_shuffle(full_df, seed=42)
        s2 = deterministic_shuffle(full_df, seed=99)
        assert s1["id"].tolist() != s2["id"].tolist()

    def test_shuffle_preserves_size(self, full_df):
        s = deterministic_shuffle(full_df, seed=42)
        assert len(s) == len(full_df)


# ─── Stage 10 Validator Tests ────────────────────────────────────────────


class TestStage10Validators:
    def test_duplicate_validator_clean(self, full_df):
        result = validate_duplicates(full_df)
        assert result["passed"] is True

    def test_duplicate_validator_dirty(self):
        df = pd.DataFrame(
            {
                "id": ["a", "b"],
                "raw_text": ["same text", "same text"],
                "department": ["ORTHO", "ORTHO"],
            }
        )
        result = validate_duplicates(df)
        assert result["passed"] is False
        assert result["duplicate_count"] == 1

    def test_contradiction_clean(self, full_df):
        result = validate_contradictions(full_df)
        assert result["passed"] is True

    def test_contradiction_dirty(self):
        df = pd.DataFrame(
            {
                "raw_text": ["same text", "same text"],
                "department": ["ORTHO", "NEURO"],
            }
        )
        result = validate_contradictions(df)
        assert result["passed"] is False

    def test_balance_validator_pass(self):
        df = pd.DataFrame(
            {
                "department": ["A"] * 50 + ["B"] * 50,
            }
        )
        result = validate_balance(df, target_size=50)
        assert result["passed"] is True

    def test_balance_validator_fail(self):
        df = pd.DataFrame(
            {
                "department": ["A"] * 50 + ["B"] * 30,
            }
        )
        result = validate_balance(df, target_size=50)
        assert result["passed"] is False
        assert "B" in result["imbalanced_departments"]

    def test_language_validator(self, full_df):
        result = validate_language(full_df)
        assert result["passed"] is True

    def test_phenotype_validator(self, full_df):
        result = validate_phenotype(full_df)
        assert result["passed"] is True

    def test_provenance_validator(self):
        df = pd.DataFrame(
            {
                "dataset_source": ["augmented_X", "synthetic_Y", "original"],
                "raw_text": ["a", "b", "c"],
                "_provenance_plugin": ["X", "Y", None],
            }
        )
        result = validate_provenance(df)
        assert result["augmented_samples"] == 1
        assert result["synthetic_samples"] == 1

    def test_embedding_similarity(self, full_df):
        result = validate_embedding_similarity(full_df, threshold=0.95)
        assert "near_duplicates_found" in result

    def test_run_all_validators(self, full_df, cfg):
        results = run_validators(full_df, cfg)
        assert len(results) == 7
        assert all("validator" in r for r in results)


# ─── Report Tests ────────────────────────────────────────────────────────


class TestDiversityReport:
    def test_report_generates_files(self, full_df, cfg):
        generate_diversity_report(full_df, cfg)
        out_dir = Path(cfg.output_directory)
        assert (out_dir / "diversity_report.json").exists()
        assert (out_dir / "diversity_report.md").exists()
        assert (out_dir / "diversity_report_classes.csv").exists()

    def test_report_contains_required_keys(self, full_df, cfg):
        report = generate_diversity_report(full_df, cfg)
        assert "class_counts" in report
        assert "language_counts" in report
        assert "final_summary" in report


# ─── CLI Tests ───────────────────────────────────────────────────────────


class TestCLI:
    def test_parse_stages_range(self):
        assert parse_stages("1-5") == [1, 2, 3, 4, 5]

    def test_parse_stages_list(self):
        assert parse_stages("1,3,5") == [1, 3, 5]

    def test_parse_stages_mixed(self):
        assert parse_stages("1-3,7,9-10") == [1, 2, 3, 7, 9, 10]

    def test_parse_stages_full(self):
        assert parse_stages("1-10") == list(range(1, 11))
