# MediTriageAI — Canonical Dataset Schema

**Specification Baseline:** v1.0.0-FROZEN  
**Gate:** GATE 2 — Canonical Dataset Reconstruction  
**Date:** 2026-08-16  
**Status:** AUTHORITATIVE  
**Governing Requirements:** FR-DATA-01, FR-DATA-02, FR-DATA-03, FR-TEXT-01 (SPEC-02)

---

## Schema Design Principles

1. Every field defined here is **required to exist** in the canonical Parquet export. Fields without values must be `NULL`, never omitted.
2. Labels are **never invented**. If a source lacks a label, the field is `NULL` with the appropriate `*_source` field set to `NULL` or `"none"`.
3. The schema is a **superset** of the historical 7-column schema (`id`, `split`, `dataset_source`, `language`, `raw_text`, `department`, `triage_level`). All existing fields are preserved; new fields extend without breaking.
4. Provenance tracking is **mandatory per SPEC-04** — both label provenance (FR-DATA-01) and text provenance (FR-TEXT-01).

---

## Canonical Schema Definition

| # | Field Name | Type | Nullable | Description |
|---|---|---|---|---|
| 1 | `sample_id` | string | **NO** | Globally unique identifier. Format: `{source_dataset}::{source_record_id}::{variant_index}`. Replaces the historical `id` field. |
| 2 | `source_dataset` | string | **NO** | Canonical name of the upstream dataset (e.g., `mtsamples`, `neiss`, `nhamcs_ed`). Matches adapter's `dataset_source` property. |
| 3 | `source_record_id` | string | **NO** | Unique identifier within the source dataset. For augmented records, this is the parent record's ID. |
| 4 | `text` | string | **NO** | The primary text field used for model input. For SOURCE records, this equals `raw_text`. For augmented records, this is the transformed text. |
| 5 | `raw_text` | string | **NO** | The original unmodified text from the source dataset. Preserved for all records including augmented ones (always points back to the SOURCE text). |
| 6 | `language` | string | **NO** | ISO 639 language code. Values: `en`, `hi`, `hi-Latn`, `hi-en`, `en-hi`. |
| 7 | `language_confidence` | string | YES | Confidence of language assignment: `native` (from source metadata), `detected` (from script/language detector), `assigned` (from augmentation pipeline). |
| 8 | `script` | string | YES | Primary script of the text: `Latin`, `Devanagari`, `Mixed`, `CJK`, `Unknown`. Determined by character-class analysis. |
| 9 | `is_code_mixed` | boolean | **NO** | `True` if the text contains significant code-mixing between two or more languages. Default `False`. |
| 10 | `provenance` | string | **NO** | Text provenance per SPEC-04 taxonomy: `SOURCE`, `A` (deterministic linguistic augmentation), `B` (rule-based/templated construction), `C` (LLM-generated). |
| 11 | `augmentation_type` | string | YES | If `provenance` ≠ `SOURCE`: the specific augmentation applied (e.g., `lexical_variation`, `phonetic_transliteration`, `hinglish_expansion`, `hard_negative`). `NULL` for SOURCE records. |
| 12 | `augmentation_parent_id` | string | YES | If augmented: the `sample_id` of the parent SOURCE record. `NULL` for SOURCE records. Must exist in the same dataset. |
| 13 | `department` | string | YES | One of 13 canonical department codes: `CARDIO_PULM`, `ED`, `ENT_OPHTHALMO`, `GEN_MED`, `GI`, `NEURO`, `OBGYN`, `ONCOLOGY_HEME`, `ORTHO`, `PEDS`, `PSYCH`, `RENAL_URO`, `SURGERY`. `NULL` if the source provides no department information. |
| 14 | `department_source` | string | YES | How the department label was derived: `native` (source dataset provides it), `mapped` (via `specialty_mapping.py` or adapter mapping table), `inferred` (via keyword/regex heuristic), `none`. |
| 15 | `department_confidence` | string | YES | Confidence: `high` (native or verified mapping), `low` (heuristic/inferred). |
| 16 | `triage_level` | string | YES | ESI severity level: `S1`, `S2`, `S3`, `S4`, `S5`, or `NULL` if no severity available. |
| 17 | `severity_source` | string | YES | How severity was derived: `native_esi` (source provides ESI), `mapped` (string-to-ESI mapping), `regex_heuristic`, `llm_generated`, `none`. |
| 18 | `split` | string | **NO** | Data split: `train`, `val`, `test`. Assigned deterministically via SHA-256 hash of `source_record_id` (not `sample_id` — ensures augmented variants stay in the same split as their parent). |
| 19 | `dataset_version` | string | **NO** | Version of the canonical dataset build. Format: `v{MAJOR}.{MINOR}.{PATCH}` (e.g., `v2.0.0`). |
| 20 | `license` | string | **NO** | SPDX license identifier or short description: `CC0-1.0`, `MIT`, `CC-BY-NC-SA-4.0`, `US-GOV-PUBLIC-DOMAIN`, etc. |
| 21 | `license_url` | string | YES | URL to the license text. |
| 22 | `source_url` | string | YES | URL to the original dataset source. |
| 23 | `quality_flags` | string | YES | Pipe-delimited quality flags: `short_text`, `truncated`, `low_confidence_dept`, `missing_severity`, `cjk_detected`, etc. `NULL` if no flags. |
| 24 | `red_flag_label` | string | YES | If the record contains red-flag clinical content: `true_positive`, `true_negative`, `hard_negative`, `unknown`. `NULL` for training records without red-flag annotation. |
| 25 | `ood_stratum` | string | YES | If the record is in an OOD evaluation stratum: stratum name (e.g., `medical_qa`, `non_triage_clinical`). `NULL` for primary training/eval records. |
| 26 | `robustness_stratum` | string | YES | If the record is in a robustness evaluation stratum: stratum name (e.g., `hinglish_variation`, `spelling_noise`, `asr_noise`). `NULL` for primary training records. |

---

## Schema Validation Rules

### Non-nullable Fields

The following fields must NEVER be `NULL`:

```
sample_id, source_dataset, source_record_id, text, raw_text,
language, is_code_mixed, provenance, split, dataset_version, license
```

### Enum Constraints

| Field | Valid Values |
|---|---|
| `language` | `en`, `hi`, `hi-Latn`, `hi-en`, `en-hi` |
| `script` | `Latin`, `Devanagari`, `Mixed`, `CJK`, `Unknown` |
| `provenance` | `SOURCE`, `A`, `B`, `C` |
| `department` | `CARDIO_PULM`, `ED`, `ENT_OPHTHALMO`, `GEN_MED`, `GI`, `NEURO`, `OBGYN`, `ONCOLOGY_HEME`, `ORTHO`, `PEDS`, `PSYCH`, `RENAL_URO`, `SURGERY` |
| `triage_level` | `S1`, `S2`, `S3`, `S4`, `S5` |
| `department_source` | `native`, `mapped`, `inferred`, `none` |
| `severity_source` | `native_esi`, `mapped`, `regex_heuristic`, `llm_generated`, `none` |
| `split` | `train`, `val`, `test` |

### Referential Integrity

1. If `provenance` ≠ `SOURCE`, then `augmentation_parent_id` must be non-null and must reference a valid `sample_id` within the same dataset build.
2. If `augmentation_parent_id` is non-null, the referenced parent must have `provenance = SOURCE`.
3. All records sharing the same `source_record_id` must be in the same `split`.

### Provenance Rules

1. `SOURCE` records: `augmentation_type = NULL`, `augmentation_parent_id = NULL`.
2. `A` records: `augmentation_type` must name the specific deterministic transformation.
3. `B` records: `augmentation_type` must name the template/rule system used.
4. `C` records: `augmentation_type` must identify the LLM and prompt strategy used.

---

## PyArrow Schema (Implementation Reference)

```python
import pyarrow as pa

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
```

---

## Migration from Historical Schema

| Historical Field | Canonical Field | Migration Notes |
|---|---|---|
| `id` | `sample_id` | Rename; format preserved but extended |
| `split` | `split` | Preserved |
| `dataset_source` | `source_dataset` | Rename for clarity |
| `language` | `language` | Preserved |
| `raw_text` | `raw_text` + `text` | `text` = `raw_text` for SOURCE records |
| `department` | `department` | Preserved |
| `triage_level` | `triage_level` | Must be validated against enum (ESI `S1`–`S5`) |
| — | `provenance` | **NEW** — must be populated for every row |
| — | `source_record_id` | **NEW** — extracted from `sample_id` |
| — | `severity_source` | **NEW** — must be populated |
| — | `department_source` | **NEW** — must be populated |
| — | `license` | **NEW** — must be populated per source |
| — | All other new fields | **NEW** — see schema definition above |
