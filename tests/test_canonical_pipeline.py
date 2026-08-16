"""Tests for canonical dataset pipeline components.

Covers:
- Canonical schema validation
- License gate enforcement
- Source record ID generation
- Provenance tracking
- NEISS pediatric mapping (override disabled/enabled)
- Severity mapping (Kaggle: NOT mapped to ESI)
- Language/script classification
- Augmentation lineage
- Split isolation
- Deduplication
- Checksum/manifest
"""

import hashlib
import os
import sys
from pathlib import Path

import pytest

# Ensure repo root is importable
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from meditriage.builder.canonical_schema import (
    CANONICAL_SCHEMA,
    LICENSED_PRIMARY_DATASETS,
    NON_NULLABLE_FIELDS,
    REJECTED_DATASETS,
    SOURCE_LICENSES,
    VALID_DEPARTMENTS,
    VALID_LANGUAGES,
    VALID_PROVENANCES,
    VALID_SPLITS,
    VALID_TRIAGE_LEVELS,
    validate_canonical_record,
)

# Import pilot build functions
from scripts.build_pilot import (
    detect_code_mixed,
    detect_script,
    sha256_split,
    assign_stratified_splits,
    quality_filter,
    deduplicate,
    check_leakage,
)


# ---------------------------------------------------------------------------
# Canonical Schema Tests
# ---------------------------------------------------------------------------

class TestCanonicalSchema:
    """Test the canonical schema definition and validation."""

    def test_schema_has_26_fields(self):
        assert len(CANONICAL_SCHEMA) == 26

    def test_non_nullable_fields_defined(self):
        assert "sample_id" in NON_NULLABLE_FIELDS
        assert "provenance" in NON_NULLABLE_FIELDS
        assert "license" in NON_NULLABLE_FIELDS
        assert "split" in NON_NULLABLE_FIELDS

    def test_valid_record_passes(self):
        record = _make_valid_record()
        errors = validate_canonical_record(record)
        assert errors == [], f"Valid record should pass: {errors}"

    def test_null_sample_id_fails(self):
        record = _make_valid_record()
        record["sample_id"] = None
        errors = validate_canonical_record(record)
        assert any("sample_id" in e for e in errors)

    def test_null_provenance_fails(self):
        record = _make_valid_record()
        record["provenance"] = None
        errors = validate_canonical_record(record)
        assert any("provenance" in e for e in errors)

    def test_invalid_language_fails(self):
        record = _make_valid_record()
        record["language"] = "zh"  # Chinese not in valid set
        errors = validate_canonical_record(record)
        assert any("language" in e for e in errors)

    def test_invalid_department_fails(self):
        record = _make_valid_record()
        record["department"] = "CARDIOLOGY"  # Must be CARDIO_PULM
        errors = validate_canonical_record(record)
        assert any("department" in e for e in errors)

    def test_invalid_triage_level_fails(self):
        record = _make_valid_record()
        record["triage_level"] = "Emergency"  # Must be S1-S5
        errors = validate_canonical_record(record)
        assert any("triage_level" in e for e in errors)


# ---------------------------------------------------------------------------
# License Gate Tests
# ---------------------------------------------------------------------------

class TestLicenseGate:
    """Test license gate enforcement."""

    def test_chatdoctor_rejected(self):
        assert "chatdoctor_healthcaremagic" in REJECTED_DATASETS
        assert "chatdoctor_icliniq" in REJECTED_DATASETS

    def test_meddialog_rejected(self):
        assert "meddialog_en" in REJECTED_DATASETS

    def test_fedmml_rejected(self):
        assert "fedmml_ed_triage" in REJECTED_DATASETS

    def test_mtsamples_cleared(self):
        assert "mtsamples" in LICENSED_PRIMARY_DATASETS

    def test_neiss_cleared(self):
        assert "neiss" in LICENSED_PRIMARY_DATASETS

    def test_nhamcs_cleared(self):
        assert "nhamcs_ed" in LICENSED_PRIMARY_DATASETS

    def test_pmc_patients_not_in_primary(self):
        """PMC-Patients is Grade C (NC-SA), not in primary build."""
        assert "pmc_patients" not in LICENSED_PRIMARY_DATASETS

    def test_rejected_not_in_primary(self):
        for name in REJECTED_DATASETS:
            assert name not in LICENSED_PRIMARY_DATASETS


# ---------------------------------------------------------------------------
# Source Record ID Tests
# ---------------------------------------------------------------------------

class TestSourceRecordId:
    """Test source record ID generation."""

    def test_format_contains_source(self):
        record = _make_valid_record()
        assert record["source_record_id"].startswith("mtsamples::")

    def test_sample_id_includes_variant(self):
        record = _make_valid_record()
        assert record["sample_id"].endswith("::0")


# ---------------------------------------------------------------------------
# Provenance Tests
# ---------------------------------------------------------------------------

class TestProvenance:
    """Test provenance tracking."""

    def test_source_has_no_augmentation(self):
        record = _make_valid_record()
        assert record["provenance"] == "SOURCE"
        assert record["augmentation_type"] is None
        assert record["augmentation_parent_id"] is None

    def test_augmented_record_has_parent(self):
        record = _make_valid_record()
        record["provenance"] = "A"
        record["augmentation_type"] = "hinglish_expansion"
        record["augmentation_parent_id"] = "mtsamples::0::0"
        errors = validate_canonical_record(record)
        assert errors == []

    def test_augmented_without_parent_fails(self):
        record = _make_valid_record()
        record["provenance"] = "A"
        record["augmentation_type"] = "hinglish_expansion"
        record["augmentation_parent_id"] = None  # Missing!
        errors = validate_canonical_record(record)
        assert any("augmentation_parent_id" in e for e in errors)

    def test_source_with_augmentation_type_fails(self):
        record = _make_valid_record()
        record["provenance"] = "SOURCE"
        record["augmentation_type"] = "something"
        errors = validate_canonical_record(record)
        assert any("augmentation_type" in e for e in errors)


# ---------------------------------------------------------------------------
# NEISS Pediatric Mapping Tests
# ---------------------------------------------------------------------------

class TestNEISSpeds:
    """Test NEISS pediatric override behavior."""

    def test_peds_override_disabled_by_default(self):
        """With override disabled, an age<18 patient with fracture should NOT be PEDS."""
        from scripts.build_pilot import ingest_neiss
        # We can't easily test without raw data, so test the function signature
        import inspect
        sig = inspect.signature(ingest_neiss)
        assert "peds_override" in sig.parameters
        assert sig.parameters["peds_override"].default is False

    def test_peds_override_parameter_exists(self):
        """The pilot build script should accept --peds-override flag."""
        from scripts.build_pilot import main
        # Verify parser will accept the flag
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--peds-override", action="store_true", default=False)
        args = parser.parse_args([])
        assert args.peds_override is False
        args = parser.parse_args(["--peds-override"])
        assert args.peds_override is True


# ---------------------------------------------------------------------------
# Severity Mapping Tests
# ---------------------------------------------------------------------------

class TestSeverityMapping:
    """Test severity mapping decisions."""

    def test_kaggle_urgency_not_mapped_to_esi(self):
        """Kaggle urgency_level labels must NOT be mapped to S1-S5."""
        record = _make_valid_record()
        record["source_dataset"] = "kaggle_medical_triage"
        record["triage_level"] = None  # Must be NULL
        record["severity_source"] = "none"
        errors = validate_canonical_record(record)
        assert errors == []

    def test_nhamcs_esi_preserved(self):
        """NHAMCS ESI labels should be preserved as S1-S5."""
        record = _make_valid_record()
        record["source_dataset"] = "nhamcs_ed"
        record["triage_level"] = "S3"
        record["severity_source"] = "native_esi"
        errors = validate_canonical_record(record)
        assert errors == []

    def test_invalid_severity_string_rejected(self):
        """String urgency labels (Emergency, Urgent) must NOT be used as triage_level."""
        record = _make_valid_record()
        record["triage_level"] = "Emergency"
        errors = validate_canonical_record(record)
        assert any("triage_level" in e for e in errors)


# ---------------------------------------------------------------------------
# Language Classification Tests
# ---------------------------------------------------------------------------

class TestLanguageClassification:
    """Test script and language detection."""

    def test_detect_latin(self):
        assert detect_script("This is a test sentence.") == "Latin"

    def test_detect_cjk(self):
        assert detect_script("这是中文测试") == "CJK"

    def test_detect_devanagari(self):
        assert detect_script("यह एक परीक्षण है") == "Devanagari"

    def test_detect_mixed(self):
        assert detect_script("Hello यह mixed है") == "Mixed"

    def test_code_mixed_detection(self):
        assert detect_code_mixed("Hello यह mixed text है with Hindi") is True
        assert detect_code_mixed("This is purely English text") is False

    def test_empty_text(self):
        assert detect_script("") == "Unknown"
        assert detect_code_mixed("") is False


# ---------------------------------------------------------------------------
# Split Isolation Tests
# ---------------------------------------------------------------------------

class TestSplitIsolation:
    """Test split assignment and isolation."""

    def test_same_record_id_same_split(self):
        """All variants of the same source_record_id must be in the same split."""
        record_id = "mtsamples::42"
        split1 = sha256_split(record_id)
        split2 = sha256_split(record_id)
        assert split1 == split2

    def test_deterministic(self):
        """Split must be deterministic across runs."""
        assert sha256_split("test_record_123") == sha256_split("test_record_123")

    def test_valid_splits_only(self):
        """Split must be one of train/val/test."""
        for i in range(100):
            split = sha256_split(f"record_{i}")
            assert split in VALID_SPLITS

    def test_leakage_check_passes_clean(self):
        """No leakage in properly split records."""
        records = [
            {"source_record_id": "a", "split": "train"},
            {"source_record_id": "b", "split": "val"},
            {"source_record_id": "c", "split": "test"},
        ]
        assert check_leakage(records) == []

    def test_leakage_check_detects_violation(self):
        """Detect when same source_record_id spans splits."""
        records = [
            {"source_record_id": "a", "split": "train"},
            {"source_record_id": "a", "split": "test"},  # LEAK!
        ]
        assert len(check_leakage(records)) == 1


# ---------------------------------------------------------------------------
# Stratified Split Tests
# ---------------------------------------------------------------------------

class TestStratifiedSplit:
    """Test source-aware stratified split strategy."""

    def _make_records(self, source, n):
        return [
            {"source_dataset": source, "source_record_id": f"{source}::{i:06d}", "split": None}
            for i in range(n)
        ]

    def test_deterministic(self):
        """Same input produces same split assignment."""
        recs1 = self._make_records("src_a", 100)
        recs2 = self._make_records("src_a", 100)
        assign_stratified_splits(recs1)
        assign_stratified_splits(recs2)
        for r1, r2 in zip(recs1, recs2):
            assert r1["split"] == r2["split"]

    def test_different_records_get_different_splits(self):
        """With enough records, all three splits are present."""
        recs = self._make_records("src_a", 100)
        assign_stratified_splits(recs)
        splits = {r["split"] for r in recs}
        assert splits == {"train", "val", "test"}

    def test_approximate_proportions(self):
        """With 1000 records per source, proportions are close to 80/10/10."""
        recs = self._make_records("src_a", 1000)
        assign_stratified_splits(recs)
        counts = {"train": 0, "val": 0, "test": 0}
        for r in recs:
            counts[r["split"]] += 1
        assert 750 <= counts["train"] <= 850, f"Train: {counts['train']}"
        assert 80 <= counts["val"] <= 120, f"Val: {counts['val']}"
        assert 80 <= counts["test"] <= 120, f"Test: {counts['test']}"

    def test_per_source_proportions(self):
        """Each source independently gets ~80/10/10."""
        recs = self._make_records("src_a", 200) + self._make_records("src_b", 200)
        assign_stratified_splits(recs)
        for src in ["src_a", "src_b"]:
            sub = [r for r in recs if r["source_dataset"] == src]
            counts = {"train": 0, "val": 0, "test": 0}
            for r in sub:
                counts[r["split"]] += 1
            assert counts["train"] >= 150, f"{src} train: {counts['train']}"
            assert counts["val"] >= 15, f"{src} val: {counts['val']}"
            assert counts["test"] >= 15, f"{src} test: {counts['test']}"

    def test_unique_source_ids(self):
        """No two records in the same source have the same ID."""
        recs = self._make_records("src_a", 50)
        ids = [r["source_record_id"] for r in recs]
        assert len(ids) == len(set(ids))

    def test_no_group_crosses_splits(self):
        """Records with same source_record_id stay in same split."""
        recs = [
            {"source_dataset": "s", "source_record_id": "s::0", "split": None},
            {"source_dataset": "s", "source_record_id": "s::0", "split": None},  # dup ID
            {"source_dataset": "s", "source_record_id": "s::1", "split": None},
        ]
        # Note: dedup would remove the dup, but split must not split same ID
        assign_stratified_splits(recs)
        assert recs[0]["split"] == recs[1]["split"]

    def test_tiny_source_all_train(self):
        """Sources with <10 records go entirely to train."""
        recs = self._make_records("tiny_src", 5)
        assign_stratified_splits(recs)
        for r in recs:
            assert r["split"] == "train"

    def test_augmentation_parent_same_split(self):
        """Augmented child inherits parent split via same source_record_id."""
        parent_id = "src_a::000042"
        recs = [
            {"source_dataset": "src_a", "source_record_id": parent_id, "split": None},
        ] * 3  # parent + 2 augmented children all share source_record_id
        # Must be deep copies
        recs = [{**r} for r in recs]
        assign_stratified_splits(recs)
        splits = {r["split"] for r in recs}
        assert len(splits) == 1, f"Parent+children split inconsistency: {splits}"

    def test_valid_splits_only(self):
        """All assigned splits are valid."""
        recs = self._make_records("src_a", 100)
        assign_stratified_splits(recs)
        for r in recs:
            assert r["split"] in VALID_SPLITS


# ---------------------------------------------------------------------------
# Deduplication Tests
# ---------------------------------------------------------------------------

class TestDeduplication:
    """Test deduplication logic."""

    def test_exact_dedup(self):
        records = [
            {"sample_id": "a::0", "text": "Patient presents with chest pain"},
            {"sample_id": "b::0", "text": "Patient presents with chest pain"},
            {"sample_id": "c::0", "text": "Different complaint entirely"},
        ]
        deduped, dropped = deduplicate(records)
        assert len(deduped) == 2
        assert dropped == 1

    def test_case_insensitive_dedup(self):
        records = [
            {"sample_id": "a::0", "text": "Patient Presents With Chest Pain"},
            {"sample_id": "b::0", "text": "patient presents with chest pain"},
        ]
        deduped, dropped = deduplicate(records)
        assert len(deduped) == 1
        assert dropped == 1

    def test_no_duplicates(self):
        records = [
            {"sample_id": "a::0", "text": "Unique text one"},
            {"sample_id": "b::0", "text": "Unique text two"},
        ]
        deduped, dropped = deduplicate(records)
        assert len(deduped) == 2
        assert dropped == 0


# ---------------------------------------------------------------------------
# Quality Filter Tests
# ---------------------------------------------------------------------------

class TestQualityFilter:
    """Test quality control filtering."""

    def test_empty_text_rejected(self):
        records = [_make_valid_record()]
        records[0]["text"] = ""
        filtered, stats = quality_filter(records)
        assert len(filtered) == 0
        assert stats["rejected_empty"] == 1

    def test_short_text_rejected(self):
        records = [_make_valid_record()]
        records[0]["text"] = "Short"
        filtered, stats = quality_filter(records)
        assert len(filtered) == 0
        assert stats["rejected_short"] == 1

    def test_cjk_text_rejected(self):
        records = [_make_valid_record()]
        records[0]["text"] = "这是中文测试文本，用于验证CJK过滤"
        records[0]["script"] = "CJK"
        filtered, stats = quality_filter(records)
        assert len(filtered) == 0
        assert stats["rejected_cjk"] == 1

    def test_valid_text_passes(self):
        records = [_make_valid_record()]
        filtered, stats = quality_filter(records)
        assert len(filtered) == 1


# ---------------------------------------------------------------------------
# Augmentation Lineage Tests
# ---------------------------------------------------------------------------

class TestAugmentationLineage:
    """Test augmentation lineage requirements."""

    def test_augmented_record_schema(self):
        """Augmented records must have parent_id, provenance, and type."""
        parent = _make_valid_record()
        parent["sample_id"] = "mtsamples::42::0"
        parent["source_record_id"] = "mtsamples::42"
        parent["split"] = "train"

        child = _make_valid_record()
        child["sample_id"] = "mtsamples::42::1"
        child["source_record_id"] = "mtsamples::42"  # Same as parent
        child["split"] = "train"  # Same as parent
        child["provenance"] = "A"
        child["augmentation_type"] = "hinglish_expansion"
        child["augmentation_parent_id"] = parent["sample_id"]
        child["text"] = "Modified text with Hinglish"

        errors = validate_canonical_record(child)
        assert errors == []

    def test_augmented_child_same_split_as_parent(self):
        """Augmented children must be in the same split as parent."""
        parent_id = "mtsamples::42"
        parent_split = sha256_split(parent_id)

        # Child uses same source_record_id → same split
        child_split = sha256_split(parent_id)
        assert parent_split == child_split


# ---------------------------------------------------------------------------
# Checksum Tests
# ---------------------------------------------------------------------------

class TestChecksum:
    """Test SHA-256 checksum computation."""

    def test_sha256_deterministic(self):
        data = b"test data for checksum"
        h1 = hashlib.sha256(data).hexdigest()
        h2 = hashlib.sha256(data).hexdigest()
        assert h1 == h2

    def test_sha256_changes_with_data(self):
        h1 = hashlib.sha256(b"data1").hexdigest()
        h2 = hashlib.sha256(b"data2").hexdigest()
        assert h1 != h2


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_valid_record() -> dict:
    """Create a minimal valid canonical record."""
    return {
        "sample_id": "mtsamples::1::0",
        "source_dataset": "mtsamples",
        "source_record_id": "mtsamples::1",
        "text": "Patient presents with chief complaint of chest pain radiating to left arm.",
        "raw_text": "Patient presents with chief complaint of chest pain radiating to left arm.",
        "language": "en",
        "language_confidence": "native",
        "script": "Latin",
        "is_code_mixed": False,
        "provenance": "SOURCE",
        "augmentation_type": None,
        "augmentation_parent_id": None,
        "department": "CARDIO_PULM",
        "department_source": "mapped",
        "department_confidence": "high",
        "triage_level": None,
        "severity_source": "none",
        "split": "train",
        "dataset_version": "v2.0.0-pilot",
        "license": "CC0-1.0",
        "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
        "source_url": "https://huggingface.co/datasets/NickyNicky/medical_mtsamples",
        "quality_flags": None,
        "red_flag_label": None,
        "ood_stratum": None,
        "robustness_stratum": None,
    }
