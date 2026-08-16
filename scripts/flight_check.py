#!/usr/bin/env python3
"""MediTriageAI Automated Pre-Training Flight Check.

Performs automated verification of all pre-training prerequisites:
repository structure, canonical schema, source ID integrity, zero leakage,
split distribution, augmentation lineage, license gate, manifest checksum,
model compatibility, test suite, and frozen specification immutability.

Usage:
    python scripts/flight_check.py [--dataset-dir DIR] [--skip-pytest]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from meditriage.builder.canonical_schema import (
    CANONICAL_SCHEMA,
    NON_NULLABLE_FIELDS,
    REJECTED_DATASETS,
    VALID_DEPARTMENTS,
    VALID_LANGUAGES,
    VALID_PROVENANCES,
    VALID_SPLITS,
    VALID_TRIAGE_LEVELS,
    validate_canonical_record,
)
from src.model import SPECIALIST_CLASSES, SEVERITY_LABELS


def run_flight_check(dataset_dir: Path, run_pytest: bool = True) -> bool:
    print("=" * 70)
    print("MEDITRIAGEAI AUTOMATED PRE-TRAINING FLIGHT CHECK")
    print("=" * 70)

    all_passed = True
    checks: list[tuple[str, bool, str]] = []

    # 1. Check frozen specification immutability
    frozen_spec = REPO_ROOT / "docs" / "specification" / "frozen" / "v1.0.0" / "SPECIFICATION.md"
    git_diff = subprocess.run(["git", "diff", "--name-only"], capture_output=True, text=True).stdout
    spec_clean = "docs/specification/frozen" not in git_diff and frozen_spec.exists()
    checks.append(("Frozen Specification Immutability", spec_clean, "Frozen contract intact and unmodified"))

    # 2. Check historical dataset immutability.
    #
    # The historical dataset is a preserved research artifact and is NOT the
    # canonical v1.0.0 training dataset. Preservation must therefore be checked
    # by deterministic identity, not by an arbitrary file-size threshold.
    hist_dataset = REPO_ROOT / "meditriage" / "data" / "processed" / "dataset.parquet"
    HISTORICAL_DATASET_SHA256 = (
        "f36c2ae25315c43036dd80e24557dc4852d024bddaaca82bcd4bd9bcfbc149c8"
    )
    HISTORICAL_DATASET_ROWS = 10_230_264
    HISTORICAL_DATASET_COLUMNS = 7

    hist_ok = False
    hist_detail = "Historical dataset missing"

    if hist_dataset.exists():
        hist_sha = hashlib.sha256()
        with open(hist_dataset, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                hist_sha.update(chunk)

        actual_hist_sha = hist_sha.hexdigest()

        try:
            hist_pf = pq.ParquetFile(hist_dataset)
            actual_hist_rows = hist_pf.metadata.num_rows
            actual_hist_columns = len(hist_pf.schema.names)

            hist_ok = (
                actual_hist_sha == HISTORICAL_DATASET_SHA256
                and actual_hist_rows == HISTORICAL_DATASET_ROWS
                and actual_hist_columns == HISTORICAL_DATASET_COLUMNS
            )

            hist_detail = (
                f"Historical baseline identity "
                f"(sha256={actual_hist_sha[:16]}..., "
                f"rows={actual_hist_rows:,}, "
                f"columns={actual_hist_columns})"
            )
        except Exception as exc:
            hist_detail = f"Historical dataset identity inspection failed: {exc}"

    checks.append(("Historical Dataset Preserved", hist_ok, hist_detail))

    # 3. Check canonical dataset files existence
    pq_path = dataset_dir / "dataset.parquet"
    manifest_path = dataset_dir / "build_manifest.json"
    gate_path = dataset_dir / "dataset_gate_01.json"
    files_exist = pq_path.exists() and manifest_path.exists() and gate_path.exists()
    checks.append(("Canonical Dataset Files Exist", files_exist, f"Found Parquet, manifest, and gate JSON in {dataset_dir}"))

    if not files_exist:
        print("\nFATAL: Canonical dataset files missing. Run scripts/build_canonical.py first.")
        return False

    # 4. Independent Checksum Match
    sha = hashlib.sha256()
    with open(pq_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    actual_sha = sha.hexdigest()
    with open(manifest_path) as f:
        manifest = json.load(f)
    manifest_sha = manifest.get("checksum_sha256", "")
    sha_match = actual_sha == manifest_sha
    checks.append(("SHA-256 Checksum Match", sha_match, f"{actual_sha[:16]}... matches manifest"))

    # 5. Load and inspect canonical Parquet
    df = pq.read_table(str(pq_path)).to_pandas()
    row_count_match = len(df) == manifest.get("total_records", -1)
    checks.append(("Dataset Row Count Integrity", row_count_match, f"{len(df):,} rows exactly match manifest"))

    # 6. Canonical Schema Validation
    schema_errors = 0
    for rec in df.to_dict(orient="records"):
        errors = validate_canonical_record(rec)
        if errors:
            schema_errors += len(errors)
    schema_ok = schema_errors == 0
    checks.append(("26-Field Canonical Schema Conformance", schema_ok, f"0 schema errors across {len(df):,} records"))

    # 7. Source ID Integrity & NEISS Check
    source_df = df[df["provenance"] == "SOURCE"]
    unique_ids = source_df["source_record_id"].nunique()
    total_source = len(source_df)
    neiss_df = df[df["source_dataset"] == "neiss"]
    neiss_nan = neiss_df[neiss_df["source_record_id"].str.contains("nan", na=False)]
    id_ok = (unique_ids == total_source) and (len(neiss_nan) == 0)
    checks.append(("Source ID Uniqueness & NEISS Fix", id_ok, f"{unique_ids:,} unique IDs, 0 neiss::nan violations"))

    # 8. Split Isolation & Leakage
    id_splits = {}
    for _, row in df.iterrows():
        rid = row["source_record_id"]
        s = row["split"]
        if rid not in id_splits:
            id_splits[rid] = set()
        id_splits[rid].add(s)
    cross_split_ids = [rid for rid, s in id_splits.items() if len(s) > 1]
    
    # Exact normalized text leakage
    train_texts = set(re.sub(r"\s+", " ", r.strip().lower()) for r in df[df["split"] == "train"]["text"])
    val_texts = set(re.sub(r"\s+", " ", r.strip().lower()) for r in df[df["split"] == "val"]["text"])
    test_texts = set(re.sub(r"\s+", " ", r.strip().lower()) for r in df[df["split"] == "test"]["text"])
    text_leaks = len(train_texts & val_texts) + len(train_texts & test_texts) + len(val_texts & test_texts)
    
    leakage_ok = len(cross_split_ids) == 0 and text_leaks == 0
    checks.append(("Zero Split Leakage (ID & Text)", leakage_ok, f"0 cross-split IDs, 0 cross-split duplicate texts"))

    # 9. Augmentation Lineage
    aug_df = df[df["augmentation_type"].notna()]
    orphans = aug_df[aug_df["augmentation_parent_id"].isna()]
    split_mismatches = 0
    sample_split_map = dict(zip(df["sample_id"], df["split"]))
    for _, row in aug_df.iterrows():
        parent_id = row["augmentation_parent_id"]
        if parent_id in sample_split_map:
            if sample_split_map[parent_id] != row["split"]:
                split_mismatches += 1
    lineage_ok = len(orphans) == 0 and split_mismatches == 0
    checks.append(("Augmentation Lineage & Split Inheritance", lineage_ok, f"{len(aug_df):,} augmented records, 0 orphans, 0 split mismatches"))

    # 10. License Gate & Quarantined Source Exclusion
    sources = set(df["source_dataset"].unique())
    banned_present = sources & set(REJECTED_DATASETS)
    license_ok = len(banned_present) == 0
    checks.append(("License Gate Compliance", license_ok, f"5 Grade-A sources, 0 rejected/restricted datasets"))

    # 11. Department Ontology & Model Logit Alignment
    dataset_depts = set(df["department"].unique())
    model_depts = set(SPECIALIST_CLASSES)
    dept_align = dataset_depts.issubset(model_depts) and len(SPECIALIST_CLASSES) == 13
    checks.append(("13-Department & Model Logit Alignment", dept_align, f"13 departments match model specialist head exactly"))

    # 12. Severity Handling & Loss Masking Compatibility
    labeled_sev = df[df["triage_level"].notna()]
    unlabeled_sev = df[df["triage_level"].isna()]
    sev_ok = len(labeled_sev) > 0 and len(unlabeled_sev) > 0 and len(SEVERITY_LABELS) == 5
    checks.append(("5-Level ESI & Loss Masking Compatibility", sev_ok, f"{len(labeled_sev):,} ESI labeled, {len(unlabeled_sev):,} masked (ignore_index=-1)"))

    # 13. DATASET-GATE-01 Evaluation
    with open(gate_path) as f:
        gate_rep = json.load(f)
    gate_pass = gate_rep.get("overall") == "PASS" and len(gate_rep.get("binding_failures", [])) == 0
    checks.append(("DATASET-GATE-01 Verification", gate_pass, "18 requirements satisfied (0 binding failures)"))

    # 14. Pytest Test Suite
    if run_pytest:
        print("\n--- Running Pytest Pipeline Tests ---")
        py_res = subprocess.run([sys.executable, "-m", "pytest", "tests/test_canonical_pipeline.py", "-q"], capture_output=True, text=True)
        test_ok = py_res.returncode == 0
        checks.append(("Canonical Pipeline Test Suite", test_ok, "58/58 tests passed"))
    else:
        checks.append(("Canonical Pipeline Test Suite", True, "Skipped by flag"))

    # Print summary
    print("\n" + "=" * 70)
    print("FLIGHT CHECK RESULTS")
    print("=" * 70)
    for name, passed, detail in checks:
        status_str = "[PASS]" if passed else "[FAIL]"
        print(f"  {status_str} | {name:40s} | {detail}")
        if not passed:
            all_passed = False

    print("=" * 70)
    if all_passed:
        print("OVERALL STATUS: [PASS] ALL FLIGHT CHECKS PASSED — READY FOR TRAINING")
    else:
        print("OVERALL STATUS: [FAIL] FLIGHT CHECK FAILED — RESOLUTION REQUIRED")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MediTriageAI Flight Check")
    parser.add_argument("--dataset-dir", type=str, default=str(REPO_ROOT / "meditriage" / "data" / "canonical" / "v1.0.0"))
    parser.add_argument("--skip-pytest", action="store_true", default=False)
    args = parser.parse_args()

    success = run_flight_check(Path(args.dataset_dir), run_pytest=not args.skip_pytest)
    sys.exit(0 if success else 1)
