#!/usr/bin/env python3
"""MediTriageAI Canonical Dataset Pilot Build.

Builds a small representative pilot dataset exercising all pipeline stages
against the canonical 26-field schema. Does NOT build the full dataset.

Usage:
    python scripts/build_pilot.py [--output-dir DIR] [--seed INT]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Resolve repo root
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent

import sys
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATASET_VERSION = "v2.0.0-pilot"
PILOT_SAMPLES_PER_SOURCE = 100  # rows per source for pilot
PILOT_NEISS_SAMPLE = 1000       # larger sample for NEISS

# Canonical department mapping for Kaggle specialties
KAGGLE_SPECIALTY_TO_DEPT = {
    "cardiology": "CARDIO_PULM",
    "emergency medicine": "ED",
    "neurology": "NEURO",
    "dermatology": "ENT_OPHTHALMO",
    "orthopedics": "ORTHO",
    "gastroenterology": "GI",
    "pulmonology": "CARDIO_PULM",
    "mental health": "PSYCH",
    "urology": "RENAL_URO",
    "endocrinology": "GEN_MED",
    "ophthalmology": "ENT_OPHTHALMO",
    "ent": "ENT_OPHTHALMO",
    "rheumatology": "ORTHO",
    "oncology": "ONCOLOGY_HEME",
}

# MTSamples specialty mapping (inlined from src/specialty_mapping.py)
MTSAMPLES_RAW_TO_DEPARTMENT = {
    "emergency room reports": "ED",
    "cardiovascular / pulmonary": "CARDIO_PULM",
    "sleep medicine": "CARDIO_PULM",
    "gastroenterology": "GI",
    "bariatrics": "GI",
    "diets and nutritions": "GI",
    "neurology": "NEURO",
    "neurosurgery": "NEURO",
    "orthopedic": "ORTHO",
    "physical medicine - rehab": "ORTHO",
    "podiatry": "ORTHO",
    "chiropractic": "ORTHO",
    "surgery": "SURGERY",
    "cosmetic / plastic surgery": "SURGERY",
    "obstetrics / gynecology": "OBGYN",
    "pediatrics - neonatal": "PEDS",
    "psychiatry / psychology": "PSYCH",
    "hematology - oncology": "ONCOLOGY_HEME",
    "nephrology": "RENAL_URO",
    "urology": "RENAL_URO",
    "ent - otolaryngology": "ENT_OPHTHALMO",
    "ophthalmology": "ENT_OPHTHALMO",
    "dermatology": "ENT_OPHTHALMO",
    "allergy / immunology": "GEN_MED",
    "endocrinology": "GEN_MED",
    "general medicine": "GEN_MED",
    "internal medicine": "GEN_MED",
    "pain management": "GEN_MED",
    "rheumatology": "GEN_MED",
    "radiology": "GEN_MED",
    "lab medicine - pathology": "GEN_MED",
    "dentistry": "ENT_OPHTHALMO",
    "letters": "GEN_MED",
    "office notes": "GEN_MED",
    "discharge summary": "GEN_MED",
    "consult - history and phy.": "GEN_MED",
    "soap / chart / progress notes": "GEN_MED",
    "hospice - palliative care": "GEN_MED",
    "autopsy": "GEN_MED",
    "ime-qme-work comp etc.": "GEN_MED",
}


# ---------------------------------------------------------------------------
# Script/Language detection helpers
# ---------------------------------------------------------------------------
CJK_RE = re.compile(r'[\u4E00-\u9FFF\u3400-\u4DBF\uF900-\uFAFF]')
DEVANAGARI_RE = re.compile(r'[\u0900-\u097F]')
LATIN_RE = re.compile(r'[A-Za-z]')


def detect_script(text: str) -> str:
    """Detect primary script of text."""
    if not text:
        return "Unknown"
    cjk = len(CJK_RE.findall(text))
    dev = len(DEVANAGARI_RE.findall(text))
    lat = len(LATIN_RE.findall(text))
    total = cjk + dev + lat
    if total == 0:
        return "Unknown"
    if cjk / total > 0.05:
        return "CJK"
    if dev > 0 and lat > 0:
        return "Mixed"
    if dev > 0:
        return "Devanagari"
    return "Latin"


def detect_code_mixed(text: str) -> bool:
    """Detect if text contains significant code-mixing."""
    if not text:
        return False
    dev = len(DEVANAGARI_RE.findall(text))
    lat = len(LATIN_RE.findall(text))
    total = dev + lat
    if total < 10:
        return False
    return dev / total > 0.1 and lat / total > 0.1


def sha256_sort_key(record_id: str) -> float:
    """Deterministic sort key via SHA-256 hash for reproducible ordering."""
    h = hashlib.sha256(record_id.encode("utf-8")).hexdigest()
    return int(h[:16], 16) / 0xFFFFFFFFFFFFFFFF


def assign_stratified_splits(
    records: list[dict],
    train_pct: float = 0.80,
    val_pct: float = 0.10,
    min_source_for_stratification: int = 10,
) -> list[dict]:
    """Source-aware deterministic stratified split assignment.

    Algorithm:
    1. Group records by source_dataset.
    2. Within each source, sort records by SHA-256 hash of source_record_id
       (deterministic, reproducible ordering).
    3. Assign first ~80% to train, next ~10% to val, remainder to test.
    4. For sources with fewer than min_source_for_stratification records,
       assign all to train (documented fallback: tiny sources cannot
       meaningfully contribute to val/test splits).
    5. Augmented records inherit their parent's split (not applied here;
       enforced at augmentation time).

    Returns the same records list with 'split' field updated.
    """
    from collections import defaultdict

    # Group by source
    source_groups: dict[str, list[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        source_groups[rec["source_dataset"]].append(i)

    for source, indices in source_groups.items():
        n = len(indices)

        # Tiny source fallback: all to train
        if n < min_source_for_stratification:
            for idx in indices:
                records[idx]["split"] = "train"
            continue

        # Sort indices by SHA-256 hash of source_record_id (deterministic order)
        indices_sorted = sorted(
            indices,
            key=lambda i: sha256_sort_key(records[i]["source_record_id"])
        )

        # Calculate split boundaries
        n_train = max(1, round(n * train_pct))
        n_val = max(1, round(n * val_pct))
        # Ensure we don't exceed total
        if n_train + n_val >= n:
            n_val = max(1, n - n_train - 1) if n > n_train + 1 else 0

        for rank, idx in enumerate(indices_sorted):
            if rank < n_train:
                records[idx]["split"] = "train"
            elif rank < n_train + n_val:
                records[idx]["split"] = "val"
            else:
                records[idx]["split"] = "test"

    return records


# Backward-compatible wrapper for tests
def sha256_split(record_id: str, train_pct: float = 0.8, val_pct: float = 0.1) -> str:
    """Deterministic split for a single record (used in tests only)."""
    h = hashlib.sha256(record_id.encode("utf-8")).hexdigest()
    val = int(h[:8], 16) / 0xFFFFFFFF
    if val < train_pct:
        return "train"
    elif val < train_pct + val_pct:
        return "val"
    return "test"


# ---------------------------------------------------------------------------
# Per-source ingestion functions (pilot-scale, canonical schema)
# ---------------------------------------------------------------------------

def ingest_mtsamples(raw_dir: Path, max_rows: int) -> list[dict]:
    """Ingest MTSamples with canonical schema."""
    csv_path = raw_dir / "mtsamples" / "mtsamples (1).csv"
    if not csv_path.exists():
        csv_path = raw_dir / "mtsamples" / "mtsamples.csv"
    if not csv_path.exists():
        # Try JSONL
        jsonl_path = raw_dir / "mtsamples" / "train.jsonl"
        if jsonl_path.exists():
            df = pd.read_json(jsonl_path, lines=True, nrows=max_rows)
        else:
            print("  WARN: MTSamples not found, skipping")
            return []
    else:
        df = pd.read_csv(csv_path, index_col=0, nrows=max_rows)

    records = []
    lic = SOURCE_LICENSES["mtsamples"]
    for idx, row in df.iterrows():
        text = str(row.get("transcription", "")).strip()
        if not text or text.lower() == "nan":
            text = str(row.get("description", "")).strip()
        if not text or text.lower() == "nan":
            continue

        specialty = str(row.get("medical_specialty", "")).strip().lower()
        if specialty == "nan" or not specialty:
            continue

        department = MTSAMPLES_RAW_TO_DEPARTMENT.get(specialty, "GEN_MED")
        record_id = f"mtsamples::{idx}"

        records.append({
            "sample_id": f"{record_id}::0",
            "source_dataset": "mtsamples",
            "source_record_id": record_id,
            "text": text,
            "raw_text": text,
            "language": "en",
            "language_confidence": "native",
            "script": detect_script(text),
            "is_code_mixed": False,
            "provenance": "SOURCE",
            "augmentation_type": None,
            "augmentation_parent_id": None,
            "department": department,
            "department_source": "mapped",
            "department_confidence": "high",
            "triage_level": None,
            "severity_source": "none",
            "split": None,  # assigned post-ingestion by assign_stratified_splits()
            "dataset_version": DATASET_VERSION,
            "license": lic["license"],
            "license_url": lic["license_url"],
            "source_url": lic["source_url"],
            "quality_flags": None,
            "red_flag_label": None,
            "ood_stratum": None,
            "robustness_stratum": None,
        })
    return records


def ingest_neiss(raw_dir: Path, max_rows: int, peds_override: bool = False) -> list[dict]:
    """Ingest NEISS with canonical schema. PEDS override configurable (default: disabled)."""
    parquet_path = raw_dir / "neiss" / "neiss_all.parquet"
    if not parquet_path.exists():
        print("  WARN: NEISS not found, skipping")
        return []

    pf = pq.ParquetFile(parquet_path)
    records = []
    lic = SOURCE_LICENSES["neiss"]
    row_idx = 0

    for batch in pf.iter_batches(batch_size=max_rows):
        chunk_df = batch.to_pandas()
        chunk_df["Narrative_1"] = chunk_df["Narrative_1"].astype(str).str.strip()
        valid_mask = (chunk_df["Narrative_1"] != "") & (chunk_df["Narrative_1"].str.lower() != "nan")
        valid_df = chunk_df[valid_mask].copy()

        if len(valid_df) == 0:
            continue

        for _, row in valid_df.head(max_rows - len(records)).iterrows():
            text = row["Narrative_1"]
            # Use CPSC_Case_Number if valid; fallback to deterministic row index
            cpsc_raw = row.get("CPSC_Case_Number")
            if pd.notna(cpsc_raw) and str(cpsc_raw).strip() not in ("", "nan", "NaN"):
                cpsc_id = str(int(cpsc_raw))  # normalize to integer string
            else:
                cpsc_id = f"row:{row_idx:08d}"  # deterministic fallback
            record_id = f"neiss::{cpsc_id}"
            row_idx += 1

            # Department heuristic (from existing adapter, without PEDS override)
            diag_code = pd.to_numeric(row.get("Diagnosis"), errors="coerce") if "Diagnosis" in row.index else float("nan")
            body_code = pd.to_numeric(row.get("Body_Part"), errors="coerce") if "Body_Part" in row.index else float("nan")

            department = "GEN_MED"
            # Diagnosis-based (check for NaN before comparing)
            if pd.notna(diag_code):
                if diag_code in (55, 57, 64): department = "ORTHO"
                elif diag_code in (52, 61): department = "NEURO"
                elif diag_code in (65, 67, 68): department = "CARDIO_PULM"
                elif diag_code in (66,): department = "GI"
                elif diag_code in (54, 58, 59): department = "ENT_OPHTHALMO"
                elif diag_code in (50, 63): department = "SURGERY"

            # Body-part refinement for unmapped
            if department == "GEN_MED" and pd.notna(body_code):
                if body_code in (76, 77): department = "ENT_OPHTHALMO"
                elif body_code in (75,): department = "NEURO"
                elif body_code in (31,): department = "CARDIO_PULM"
                elif body_code in (30, 34, 35, 36, 37): department = "ORTHO"
                elif body_code in (33, 38): department = "RENAL_URO"

            # Narrative-based refinement for remaining GEN_MED
            if department == "GEN_MED":
                text_lower = text.lower()
                if re.search(r"fracture|sprain|strain|bone|joint|knee|shoulder|ankle|wrist|hip|dislocation", text_lower):
                    department = "ORTHO"
                elif re.search(r"head injury|concussion|headache|dizziness|seizure|loss of consciousness", text_lower):
                    department = "NEURO"
                elif re.search(r"chest pain|shortness of breath|asthma|heart|lung|breathing", text_lower):
                    department = "CARDIO_PULM"
                elif re.search(r"eye|cornea|vision|ear|nose|throat|swallowed", text_lower):
                    department = "ENT_OPHTHALMO"
                elif re.search(r"laceration|cut|burn|rash|skin|abrasion", text_lower):
                    department = "ENT_OPHTHALMO"

            # Configurable PEDS override (default: DISABLED)
            if peds_override:
                try:
                    age = pd.to_numeric(row.get("Age"), errors="coerce")
                    if pd.notna(age) and age < 18:
                        department = "PEDS"
                except (ValueError, TypeError):
                    pass

            records.append({
                "sample_id": f"{record_id}::0",
                "source_dataset": "neiss",
                "source_record_id": record_id,
                "text": text,
                "raw_text": text,
                "language": "en",
                "language_confidence": "native",
                "script": detect_script(text),
                "is_code_mixed": False,
                "provenance": "SOURCE",
                "augmentation_type": None,
                "augmentation_parent_id": None,
                "department": department,
                "department_source": "inferred",
                "department_confidence": "low",
                "triage_level": None,
                "severity_source": "none",
                "split": None,  # assigned post-ingestion
                "dataset_version": DATASET_VERSION,
                "license": lic["license"],
                "license_url": lic["license_url"],
                "source_url": lic["source_url"],
                "quality_flags": None,
                "red_flag_label": None,
                "ood_stratum": None,
                "robustness_stratum": None,
            })

            if len(records) >= max_rows:
                break
        if len(records) >= max_rows:
            break

    return records


def ingest_nhamcs(raw_dir: Path, max_rows: int) -> list[dict]:
    """Ingest NHAMCS ED with canonical schema. Preserves genuine ESI triage levels."""
    import json as _json

    dict_path = REPO_ROOT / "meditriage" / "builder" / "adapters" / "nhamcs_dict.json"
    if not dict_path.exists():
        print("  WARN: NHAMCS dict not found, skipping")
        return []
    with open(dict_path) as f:
        col_dicts = _json.load(f)

    records = []
    lic = SOURCE_LICENSES["nhamcs_ed"]
    row_idx = 0

    for year in ["2019", "2020", "2021"]:
        year_path = raw_dir / "nhamcs_ed" / f"ed{year}"
        data_file = year_path / f"ed{year}"
        if not data_file.exists():
            continue
        cols = col_dicts.get(year)
        if not cols:
            continue

        with open(data_file, "r", encoding="ascii", errors="ignore") as f:
            for line in f:
                parsed = {}
                for col in cols:
                    parsed[col["name"]] = line[col["start"]:col["start"] + col["length"]].strip()

                # ESI triage level
                triage = parsed.get("IMMEDR", "")
                triage_level = None
                if triage in ("1", "2", "3", "4", "5"):
                    triage_level = f"S{triage}"
                elif triage in ("01", "02", "03", "04", "05"):
                    triage_level = f"S{int(triage)}"

                rfv1 = parsed.get("RFV1", "")
                rfv2 = parsed.get("RFV2", "")
                rfv3 = parsed.get("RFV3", "")
                age = parsed.get("AGE", "")
                sex = parsed.get("SEX", "")
                sex_str = "Female" if sex == "1" else "Male" if sex == "2" else "Unknown"

                raw_text = f"Age: {age}, Sex: {sex_str}\n"
                raw_text += f"Reason for Visit 1 (Code): {rfv1}\n"
                if rfv2 and rfv2 != "-0009":
                    raw_text += f"Reason for Visit 2 (Code): {rfv2}\n"
                if rfv3 and rfv3 != "-0009":
                    raw_text += f"Reason for Visit 3 (Code): {rfv3}\n"

                record_id = f"nhamcs_ed::{year}::{row_idx:08d}"
                row_idx += 1

                records.append({
                    "sample_id": f"{record_id}::0",
                    "source_dataset": "nhamcs_ed",
                    "source_record_id": record_id,
                    "text": raw_text,
                    "raw_text": raw_text,
                    "language": "en",
                    "language_confidence": "native",
                    "script": "Latin",
                    "is_code_mixed": False,
                    "provenance": "SOURCE",
                    "augmentation_type": None,
                    "augmentation_parent_id": None,
                    "department": "ED",
                    "department_source": "native",
                    "department_confidence": "high",
                    "triage_level": triage_level,
                    "severity_source": "native_esi" if triage_level else "none",
                    "split": None,  # assigned post-ingestion
                    "dataset_version": DATASET_VERSION,
                    "license": lic["license"],
                    "license_url": lic["license_url"],
                    "source_url": lic["source_url"],
                    "quality_flags": None,
                    "red_flag_label": None,
                    "ood_stratum": None,
                    "robustness_stratum": None,
                })

                if len(records) >= max_rows:
                    break
            if len(records) >= max_rows:
                break

    return records


def ingest_symptom2disease(raw_dir: Path, max_rows: int) -> list[dict]:
    """Ingest Symptom2Disease with canonical schema."""
    csv_path = raw_dir / "symptom2disease" / "Symptom2Disease.csv"
    if not csv_path.exists():
        print("  WARN: Symptom2Disease not found, skipping")
        return []

    df = pd.read_csv(csv_path, nrows=max_rows)
    records = []
    lic = SOURCE_LICENSES["symptom2disease"]

    dept_mapping = {
        "Psoriasis": "ENT_OPHTHALMO", "Varicose Veins": "CARDIO_PULM",
        "Typhoid": "GEN_MED", "Chicken pox": "GEN_MED",
        "Impetigo": "ENT_OPHTHALMO", "Dengue": "GEN_MED",
        "Fungal infection": "ENT_OPHTHALMO", "Common Cold": "GEN_MED",
        "Pneumonia": "CARDIO_PULM", "Dimorphic Hemorrhoids": "GI",
        "Arthritis": "ORTHO", "Acne": "ENT_OPHTHALMO",
        "Bronchial Asthma": "CARDIO_PULM", "Hypertension": "CARDIO_PULM",
        "Migraine": "NEURO", "Cervical spondylosis": "ORTHO",
        "Jaundice": "GI", "Malaria": "GEN_MED",
        "urinary tract infection": "RENAL_URO", "allergy": "ENT_OPHTHALMO",
    }

    for idx, row in df.iterrows():
        text = str(row.get("text", "")).strip()
        if not text or text.lower() == "nan":
            continue
        label = str(row.get("label", "")).strip()
        if label.lower() == "nan":
            label = None
        department = dept_mapping.get(label, "GEN_MED") if label else None
        record_id = f"symptom2disease::{idx:06d}"

        records.append({
            "sample_id": f"{record_id}::0",
            "source_dataset": "symptom2disease",
            "source_record_id": record_id,
            "text": text,
            "raw_text": text,
            "language": "en",
            "language_confidence": "native",
            "script": detect_script(text),
            "is_code_mixed": False,
            "provenance": "SOURCE",
            "augmentation_type": None,
            "augmentation_parent_id": None,
            "department": department,
            "department_source": "mapped" if department else "none",
            "department_confidence": "high" if department else None,
            "triage_level": None,
            "severity_source": "none",
            "split": None,  # assigned post-ingestion
            "dataset_version": DATASET_VERSION,
            "license": lic["license"],
            "license_url": lic["license_url"],
            "source_url": lic["source_url"],
            "quality_flags": None,
            "red_flag_label": None,
            "ood_stratum": None,
            "robustness_stratum": None,
        })
    return records


def ingest_kaggle_triage(raw_dir: Path, max_rows: int) -> list[dict]:
    """Ingest Kaggle Medical Triage with canonical schema.

    IMPORTANT: urgency_level labels (Emergency/Urgent/Routine/Observation)
    are NOT mapped to ESI S1-S5. They are preserved as-is in quality_flags
    with severity_source='none' and triage_level=NULL per governance decision.
    """
    data_dir = raw_dir / "kaggle_medical_triage" / "data"
    if not data_dir.exists():
        print("  WARN: Kaggle Medical Triage not found, skipping")
        return []

    parquet_files = sorted(data_dir.glob("*.parquet"))
    if not parquet_files:
        print("  WARN: Kaggle Medical Triage parquet files not found, skipping")
        return []

    records = []
    lic = SOURCE_LICENSES["kaggle_medical_triage"]
    row_idx = 0

    for pf in parquet_files:
        df = pd.read_parquet(pf)
        for _, row in df.head(max_rows - len(records)).iterrows():
            complaint = str(row.get("symptom_description", "")).strip()
            if not complaint:
                continue

            raw_text = f"Chief Complaint: {complaint}\n"
            demo = row.get("demographic_context")
            if pd.notna(demo):
                raw_text += f"Demographics: {demo}\n"
            duration = row.get("duration")
            if pd.notna(duration):
                raw_text += f"Duration: {duration}\n"
            reasoning = row.get("reasoning")
            if pd.notna(reasoning):
                raw_text += f"Clinical Reasoning: {reasoning}\n"

            # Department from primary_specialty
            specialty = str(row.get("primary_specialty", "")).strip().lower()
            department = KAGGLE_SPECIALTY_TO_DEPT.get(specialty, "GEN_MED")

            # Urgency label: preserved as metadata, NOT mapped to ESI
            urgency = row.get("urgency_level")
            urgency_flag = f"raw_urgency={urgency}" if pd.notna(urgency) else None

            record_id = f"kaggle_medical_triage::{row_idx:06d}"
            row_idx += 1

            records.append({
                "sample_id": f"{record_id}::0",
                "source_dataset": "kaggle_medical_triage",
                "source_record_id": record_id,
                "text": raw_text,
                "raw_text": raw_text,
                "language": "en",
                "language_confidence": "native",
                "script": detect_script(raw_text),
                "is_code_mixed": False,
                "provenance": "SOURCE",
                "augmentation_type": None,
                "augmentation_parent_id": None,
                "department": department,
                "department_source": "mapped",
                "department_confidence": "high",
                "triage_level": None,  # NOT mapped — see governance decision
                "severity_source": "none",
                "split": None,  # assigned post-ingestion
                "dataset_version": DATASET_VERSION,
                "license": lic["license"],
                "license_url": lic["license_url"],
                "source_url": lic["source_url"],
                "quality_flags": urgency_flag,
                "red_flag_label": None,
                "ood_stratum": None,
                "robustness_stratum": None,
            })

            if len(records) >= max_rows:
                break
        if len(records) >= max_rows:
            break
    return records


# ---------------------------------------------------------------------------
# Quality control
# ---------------------------------------------------------------------------

def quality_filter(records: list[dict]) -> tuple[list[dict], dict]:
    """Filter and flag records for quality. Returns (filtered, stats)."""
    accepted = []
    stats = {"total_in": len(records), "rejected_empty": 0, "rejected_short": 0,
             "rejected_cjk": 0, "flagged_long": 0, "flagged_short": 0}

    for rec in records:
        text = rec.get("text", "")
        if not text or not text.strip():
            stats["rejected_empty"] += 1
            continue
        if len(text.strip()) < 10:
            stats["rejected_short"] += 1
            continue

        # CJK safety filter
        script = rec.get("script", "Unknown")
        if script == "CJK":
            stats["rejected_cjk"] += 1
            continue

        # Flagging
        flags = rec.get("quality_flags") or ""
        if len(text) > 2000:
            flags = (flags + "|long_text").lstrip("|")
            stats["flagged_long"] += 1
        if len(text) < 50:
            flags = (flags + "|short_text").lstrip("|")
            stats["flagged_short"] += 1

        rec["quality_flags"] = flags if flags else None
        accepted.append(rec)

    stats["total_out"] = len(accepted)
    return accepted, stats


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def deduplicate(records: list[dict]) -> tuple[list[dict], int]:
    """Exact deduplication by normalized text. Returns (deduped, count_dropped)."""
    seen = {}
    deduped = []
    dropped = 0

    for rec in records:
        norm = rec["text"].strip().lower()
        text_hash = hashlib.sha256(norm.encode("utf-8")).hexdigest()
        if text_hash in seen:
            dropped += 1
            continue
        seen[text_hash] = rec["sample_id"]
        deduped.append(rec)

    return deduped, dropped


# ---------------------------------------------------------------------------
# Leakage check
# ---------------------------------------------------------------------------

def check_leakage(records: list[dict]) -> list[str]:
    """Check that no source_record_id spans multiple splits."""
    id_to_splits: dict[str, set] = {}
    for rec in records:
        rid = rec["source_record_id"]
        split = rec["split"]
        if rid not in id_to_splits:
            id_to_splits[rid] = set()
        id_to_splits[rid].add(split)

    violations = [rid for rid, splits in id_to_splits.items() if len(splits) > 1]
    return violations


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_pilot(records: list[dict]) -> dict:
    """Full schema validation of the pilot dataset."""
    results = {
        "total_records": len(records),
        "schema_errors": 0,
        "non_nullable_errors": 0,
        "enum_errors": 0,
        "provenance_errors": 0,
        "error_details": [],
    }

    for i, rec in enumerate(records):
        errors = validate_canonical_record(rec)
        if errors:
            results["schema_errors"] += len(errors)
            for e in errors:
                if "Non-nullable" in e:
                    results["non_nullable_errors"] += 1
                elif "Invalid value" in e:
                    results["enum_errors"] += 1
                elif "Provenance" in e or "augmentation" in e.lower():
                    results["provenance_errors"] += 1
            if len(results["error_details"]) < 20:
                results["error_details"].append({
                    "record_index": i,
                    "sample_id": rec.get("sample_id"),
                    "errors": errors
                })

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MediTriageAI Pilot Build")
    parser.add_argument("--output-dir", type=str,
                        default=str(REPO_ROOT / "meditriage" / "data" / "canonical" / "pilot"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--peds-override", action="store_true", default=False,
                        help="Enable PEDS age<18 override for NEISS (default: disabled)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = REPO_ROOT / "datasets" / "raw"

    start_time = time.time()
    all_records: list[dict] = []

    print("=" * 60)
    print("MediTriageAI Canonical Dataset Pilot Build")
    print(f"Version: {DATASET_VERSION}")
    print(f"Output: {output_dir}")
    print(f"PEDS override: {'ENABLED' if args.peds_override else 'DISABLED'}")
    print("=" * 60)

    # --- LICENSE GATE ---
    print("\n--- License Gate ---")
    for name in REJECTED_DATASETS:
        print(f"  REJECTED: {name} (Grade {SOURCE_LICENSES[name]['grade']})")
    for name in sorted(LICENSED_PRIMARY_DATASETS):
        print(f"  CLEARED:  {name} (Grade {SOURCE_LICENSES[name]['grade']})")

    # --- STAGE 3: Ingest ---
    print("\n--- Stage 3: Ingest ---")

    print("  Ingesting MTSamples...")
    mtsamples = ingest_mtsamples(raw_dir, PILOT_SAMPLES_PER_SOURCE)
    print(f"    -> {len(mtsamples)} records")
    all_records.extend(mtsamples)

    print("  Ingesting NEISS (pilot sample)...")
    neiss = ingest_neiss(raw_dir, PILOT_NEISS_SAMPLE, peds_override=args.peds_override)
    print(f"    -> {len(neiss)} records")
    all_records.extend(neiss)

    print("  Ingesting NHAMCS ED...")
    nhamcs = ingest_nhamcs(raw_dir, PILOT_SAMPLES_PER_SOURCE)
    print(f"    -> {len(nhamcs)} records")
    all_records.extend(nhamcs)

    print("  Ingesting Symptom2Disease...")
    s2d = ingest_symptom2disease(raw_dir, PILOT_SAMPLES_PER_SOURCE)
    print(f"    -> {len(s2d)} records")
    all_records.extend(s2d)

    print("  Ingesting Kaggle Medical Triage...")
    kaggle = ingest_kaggle_triage(raw_dir, PILOT_SAMPLES_PER_SOURCE)
    print(f"    -> {len(kaggle)} records")
    all_records.extend(kaggle)

    print(f"\n  Total ingested: {len(all_records)}")

    # --- STAGE 8: Quality Control ---
    print("\n--- Stage 8: Quality Control ---")
    all_records, qc_stats = quality_filter(all_records)
    print(f"  Rejected (empty): {qc_stats['rejected_empty']}")
    print(f"  Rejected (short): {qc_stats['rejected_short']}")
    print(f"  Rejected (CJK):   {qc_stats['rejected_cjk']}")
    print(f"  Flagged (long):    {qc_stats['flagged_long']}")
    print(f"  Flagged (short):   {qc_stats['flagged_short']}")
    print(f"  Total after QC:    {qc_stats['total_out']}")

    # --- STAGE 9: Deduplication ---
    print("\n--- Stage 9: Deduplication ---")
    all_records, dedup_dropped = deduplicate(all_records)
    print(f"  Dropped duplicates: {dedup_dropped}")
    print(f"  Total after dedup:  {len(all_records)}")

    # --- STAGE 9.5: Source Record ID Validation ---
    print("\n--- Stage 9.5: Source Record ID Validation ---")
    from collections import Counter
    id_counts = Counter(rec["source_record_id"] for rec in all_records)
    duplicated_ids = {k: v for k, v in id_counts.items() if v > 1}
    if duplicated_ids:
        print(f"  WARNING: {len(duplicated_ids)} non-unique source_record_ids!")
        for k, v in list(duplicated_ids.items())[:5]:
            print(f"    {k}: {v} occurrences")
    else:
        print(f"  All {len(id_counts)} source_record_ids are unique. PASS.")

    # --- STAGE 11: Stratified Split Assignment ---
    print("\n--- Stage 11: Stratified Split Assignment ---")
    all_records = assign_stratified_splits(all_records)

    # --- STAGE 10: Leakage Check ---
    print("\n--- Stage 10: Leakage Check ---")
    leakage_violations = check_leakage(all_records)
    if leakage_violations:
        print(f"  LEAKAGE DETECTED: {len(leakage_violations)} source_record_ids span multiple splits!")
        for v in leakage_violations[:5]:
            print(f"    {v}")
    else:
        print("  No leakage detected. PASS.")

    # --- STAGE 11: Split verification ---
    print("\n--- Stage 11: Split Distribution ---")
    split_counts = {}
    for rec in all_records:
        s = rec["split"]
        split_counts[s] = split_counts.get(s, 0) + 1
    total = len(all_records)
    for s in ["train", "val", "test"]:
        c = split_counts.get(s, 0)
        print(f"  {s}: {c} ({100*c/total:.1f}%)")

    # --- STAGE 15: Validation ---
    print("\n--- Stage 15: Schema Validation ---")
    val_results = validate_pilot(all_records)
    print(f"  Total records:       {val_results['total_records']}")
    print(f"  Schema errors:       {val_results['schema_errors']}")
    print(f"  Non-nullable errors: {val_results['non_nullable_errors']}")
    print(f"  Enum errors:         {val_results['enum_errors']}")
    print(f"  Provenance errors:   {val_results['provenance_errors']}")
    if val_results["error_details"]:
        print("  First errors:")
        for e in val_results["error_details"][:5]:
            print(f"    {e['sample_id']}: {e['errors']}")

    # --- STAGE 16: Export + Manifest + SHA-256 ---
    print("\n--- Stage 16: Export ---")
    df = pd.DataFrame(all_records)

    # Ensure column order matches schema
    schema_cols = [field.name for field in CANONICAL_SCHEMA]
    for col in schema_cols:
        if col not in df.columns:
            df[col] = None
    df = df[schema_cols]

    # Write Parquet
    out_pq = output_dir / "dataset.parquet"
    table = pa.Table.from_pandas(df, schema=CANONICAL_SCHEMA, preserve_index=False)
    pq.write_table(table, out_pq)
    print(f"  Parquet: {out_pq} ({out_pq.stat().st_size} bytes)")

    # SHA-256
    sha256 = hashlib.sha256()
    with open(out_pq, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    checksum = sha256.hexdigest()
    print(f"  SHA-256: {checksum}")

    # --- Statistics ---
    source_counts = df["source_dataset"].value_counts().to_dict()
    dept_counts = df["department"].value_counts().to_dict()
    lang_counts = df["language"].value_counts().to_dict()
    prov_counts = df["provenance"].value_counts().to_dict()
    sev_counts = df["triage_level"].dropna().value_counts().to_dict()
    sev_source_counts = df["severity_source"].value_counts().to_dict()

    # --- Language robustness coverage ---
    lang_coverage = {
        "english": len(df[df["language"] == "en"]),
        "hindi_devanagari": len(df[df["script"] == "Devanagari"]),
        "roman_hindi": len(df[df["language"] == "hi-Latn"]),
        "hinglish_code_mixed": len(df[df["is_code_mixed"] == True]),
        "status": {}
    }
    categories = [
        "english", "hindi_devanagari", "roman_hindi", "hinglish_code_mixed",
        "spelling_variation", "phonetic_variation", "informal_language",
        "clinical_shorthand", "negation", "temporal_expressions",
        "severity_modifiers", "asr_like_noise",
    ]
    for cat in categories:
        count = lang_coverage.get(cat, 0)
        lang_coverage["status"][cat] = "PRESENT" if count > 0 else "ABSENT"

    # Build manifest
    duration = time.time() - start_time
    manifest = {
        "dataset_version": DATASET_VERSION,
        "build_type": "pilot",
        "build_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "build_duration_seconds": round(duration, 2),
        "total_records": len(all_records),
        "checksum_sha256": checksum,
        "sources": source_counts,
        "departments": dept_counts,
        "languages": lang_counts,
        "provenances": prov_counts,
        "severity_distribution": sev_counts,
        "severity_source_distribution": sev_source_counts,
        "splits": split_counts,
        "quality_control": qc_stats,
        "dedup_dropped": dedup_dropped,
        "leakage_violations": len(leakage_violations),
        "schema_validation": val_results,
        "language_robustness_coverage": lang_coverage,
        "peds_override_enabled": args.peds_override,
    }

    manifest_path = output_dir / "build_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"  Manifest: {manifest_path}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("PILOT BUILD SUMMARY")
    print("=" * 60)
    print(f"  Total records: {len(all_records)}")
    print(f"  Sources: {list(source_counts.keys())}")
    print(f"  Schema errors: {val_results['schema_errors']}")
    print(f"  Leakage: {'FAIL' if leakage_violations else 'PASS'}")
    print(f"  Dedup dropped: {dedup_dropped}")
    print(f"  SHA-256: {checksum}")
    print(f"  Duration: {duration:.2f}s")
    print(f"  Output: {output_dir}")

    # Return pass/fail
    passed = (
        val_results["schema_errors"] == 0
        and len(leakage_violations) == 0
        and len(all_records) > 0
    )
    print(f"\n  PILOT: {'PASS' if passed else 'FAIL'}")
    print("=" * 60)

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
