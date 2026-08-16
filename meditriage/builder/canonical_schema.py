"""Canonical dataset schema for MediTriageAI v2.0.0.

Defines the 26-field canonical schema, validation rules, and PyArrow schema
for the canonical training dataset as specified in DATASET_SCHEMA.md.
"""

from __future__ import annotations

import pyarrow as pa

# --- Canonical PyArrow Schema ---

CANONICAL_SCHEMA = pa.schema([
    ("sample_id", pa.string()),
    ("source_dataset", pa.string()),
    ("source_record_id", pa.string()),
    ("text", pa.string()),
    ("raw_text", pa.string()),
    ("language", pa.string()),
    ("language_confidence", pa.string()),
    ("script", pa.string()),
    ("is_code_mixed", pa.bool_()),
    ("provenance", pa.string()),
    ("augmentation_type", pa.string()),
    ("augmentation_parent_id", pa.string()),
    ("department", pa.string()),
    ("department_source", pa.string()),
    ("department_confidence", pa.string()),
    ("triage_level", pa.string()),
    ("severity_source", pa.string()),
    ("split", pa.string()),
    ("dataset_version", pa.string()),
    ("license", pa.string()),
    ("license_url", pa.string()),
    ("source_url", pa.string()),
    ("quality_flags", pa.string()),
    ("red_flag_label", pa.string()),
    ("ood_stratum", pa.string()),
    ("robustness_stratum", pa.string()),
])

# --- Enum Constraints ---

VALID_LANGUAGES = {"en", "hi", "hi-Latn", "hi-en", "en-hi"}
VALID_SCRIPTS = {"Latin", "Devanagari", "Mixed", "CJK", "Unknown"}
VALID_PROVENANCES = {"SOURCE", "A", "B", "C"}
VALID_DEPARTMENTS = {
    "CARDIO_PULM", "ED", "ENT_OPHTHALMO", "GEN_MED", "GI", "NEURO",
    "OBGYN", "ONCOLOGY_HEME", "ORTHO", "PEDS", "PSYCH", "RENAL_URO",
    "SURGERY",
}
VALID_TRIAGE_LEVELS = {"S1", "S2", "S3", "S4", "S5"}
VALID_DEPARTMENT_SOURCES = {"native", "mapped", "inferred", "none"}
VALID_SEVERITY_SOURCES = {"native_esi", "mapped", "regex_heuristic", "llm_generated", "none"}
VALID_SPLITS = {"train", "val", "test"}

# Non-nullable fields
NON_NULLABLE_FIELDS = {
    "sample_id", "source_dataset", "source_record_id", "text", "raw_text",
    "language", "is_code_mixed", "provenance", "split", "dataset_version",
    "license",
}

# --- License Registry ---

SOURCE_LICENSES = {
    "mtsamples": {"license": "CC0-1.0", "license_url": "https://creativecommons.org/publicdomain/zero/1.0/", "source_url": "https://huggingface.co/datasets/NickyNicky/medical_mtsamples", "grade": "A"},
    "neiss": {"license": "US-GOV-PUBLIC-DOMAIN", "license_url": "https://www.cpsc.gov/cgibin/NEISSQuery/home.aspx", "source_url": "https://huggingface.co/datasets/Layered-Labs/neiss-injury-data", "grade": "A"},
    "nhamcs_ed": {"license": "US-GOV-PUBLIC-USE", "license_url": "https://www.cdc.gov/nchs/data_access/restrictions.htm", "source_url": "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datasets/NHAMCS/", "grade": "A"},
    "symptom2disease": {"license": "CC0-1.0", "license_url": "https://creativecommons.org/publicdomain/zero/1.0/", "source_url": "https://huggingface.co/datasets/NeuronZero/Symptom2Disease", "grade": "A"},
    "kaggle_medical_triage": {"license": "CC0-1.0", "license_url": "https://creativecommons.org/publicdomain/zero/1.0/", "source_url": "https://huggingface.co/datasets/sweatSmile/medical-symptom-triage-csv", "grade": "A"},
    "pmc_patients": {"license": "CC-BY-NC-SA-4.0", "license_url": "https://creativecommons.org/licenses/by-nc-sa/4.0/", "source_url": "https://huggingface.co/datasets/zhengyun21/PMC-Patients", "grade": "C"},
    "l3cube_code_mixed": {"license": "MIT", "license_url": "https://github.com/l3cube-pune/code-mixed-nlp/blob/main/LICENSE", "source_url": "https://raw.githubusercontent.com/l3cube-pune/code-mixed-nlp/main/L3Cube-HingLID/train.txt", "grade": "A"},
    "medqa_usmle": {"license": "MIT", "license_url": "https://github.com/jind11/MedQA", "source_url": "https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options", "grade": "A"},
    "medical_meadow_medqa": {"license": "CC-BY-4.0", "license_url": "https://creativecommons.org/licenses/by/4.0/", "source_url": "https://huggingface.co/datasets/medalpaca/medical_meadow_medqa", "grade": "A"},
    # Quarantined / Rejected
    "chatdoctor_healthcaremagic": {"license": "REJECTED", "license_url": "", "source_url": "", "grade": "E"},
    "chatdoctor_icliniq": {"license": "REJECTED", "license_url": "", "source_url": "", "grade": "E"},
    "fedmml_ed_triage": {"license": "UNKNOWN", "license_url": "", "source_url": "", "grade": "D"},
    "meddialog_en": {"license": "UNKNOWN", "license_url": "", "source_url": "", "grade": "D"},
}

# Datasets allowed in the canonical primary build (Grade A only for primary training)
LICENSED_PRIMARY_DATASETS = {
    name for name, info in SOURCE_LICENSES.items()
    if info["grade"] == "A"
}

# Datasets explicitly rejected or quarantined
REJECTED_DATASETS = {
    name for name, info in SOURCE_LICENSES.items()
    if info["grade"] in ("D", "E")
}


def _is_null(val: object) -> bool:
    """Check if a value is null (None, float NaN, or empty string)."""
    if val is None:
        return True
    if isinstance(val, float) and (val != val or str(val) == "nan"):
        return True
    if isinstance(val, str) and val.strip() == "":
        return True
    return False


def validate_canonical_record(record: dict) -> list[str]:
    """Validate a single record against the canonical schema.

    Returns a list of error messages (empty if valid).
    """
    errors = []

    # Non-nullable checks
    for field in NON_NULLABLE_FIELDS:
        val = record.get(field)
        if _is_null(val):
            errors.append(f"Non-nullable field '{field}' is null/empty")

    # Enum checks (only if value is non-null)
    enum_checks = [
        ("language", VALID_LANGUAGES),
        ("script", VALID_SCRIPTS),
        ("provenance", VALID_PROVENANCES),
        ("department", VALID_DEPARTMENTS),
        ("triage_level", VALID_TRIAGE_LEVELS),
        ("department_source", VALID_DEPARTMENT_SOURCES),
        ("severity_source", VALID_SEVERITY_SOURCES),
        ("split", VALID_SPLITS),
    ]
    for field, valid_values in enum_checks:
        val = record.get(field)
        if not _is_null(val) and val not in valid_values:
            errors.append(f"Invalid value for '{field}': {val!r} (valid: {valid_values})")

    # Provenance integrity
    prov = record.get("provenance")
    if prov == "SOURCE":
        if not _is_null(record.get("augmentation_type")):
            errors.append("SOURCE record must have augmentation_type=NULL")
        if not _is_null(record.get("augmentation_parent_id")):
            errors.append("SOURCE record must have augmentation_parent_id=NULL")
    elif prov in ("A", "B", "C"):
        if _is_null(record.get("augmentation_type")):
            errors.append(f"Provenance '{prov}' record must have augmentation_type set")
        if _is_null(record.get("augmentation_parent_id")):
            errors.append(f"Provenance '{prov}' record must have augmentation_parent_id set")

    return errors
