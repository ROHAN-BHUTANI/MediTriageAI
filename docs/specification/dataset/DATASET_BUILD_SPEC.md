# MediTriageAI — Canonical Dataset Build Specification

**Specification Baseline:** v1.0.0-FROZEN  
**Gate:** GATE 2 — Canonical Dataset Reconstruction  
**Date:** 2026-08-16  
**Status:** AUTHORITATIVE  
**Schema:** `docs/specification/dataset/DATASET_SCHEMA.md`  
**Selection:** `docs/specification/dataset/DATASET_SELECTION.md`  
**License Register:** `docs/specification/dataset/DATASET_LICENSE_REGISTER.md`

---

## Pipeline Overview

The canonical dataset build pipeline is a **deterministic, rerunnable, multi-stage process** that transforms raw source datasets into a DATASET-GATE-01-compliant canonical training corpus.

```
DISCOVERY
→ LICENSE GATE
→ RAW IMMUTABLE STORAGE
→ SOURCE MANIFEST
→ ADAPTER (per-source ingestion)
→ NORMALIZATION (schema alignment)
→ LABEL MAPPING (department + severity)
→ LANGUAGE CLASSIFICATION (script detection)
→ PROVENANCE (text + label provenance)
→ QUALITY CONTROL (filtering + flagging)
→ DEDUPLICATION (exact + near-duplicate)
→ LEAKAGE CONTROL (split-aware dedup)
→ SPLIT (deterministic, seed-based)
→ APPROVED AUGMENTATION (multilingual + variation)
→ ROBUSTNESS STRATA (robustness test set)
→ SAFETY STRATA (red-flag + OOD evaluation)
→ FINAL VALIDATION (schema + integrity checks)
→ MANIFEST (build metadata)
→ SHA-256 (checksum)
→ DATASET-GATE-01 (gate report)
```

---

## Stage Specifications

### Stage 0: DISCOVERY + LICENSE GATE

**Input:** Source URLs from `DATASET_LICENSE_REGISTER.md`  
**Output:** Verified raw data presence in `datasets/raw/{source_name}/`

- Verify each selected source exists in `datasets/raw/`
- Verify LICENSE GRADE is A (or C with explicit research-only marking)
- **REJECT** any source with Grade D or E
- **QUARANTINE** any source with Grade D pending investigation
- Record `SOURCE_URL.txt` in each raw directory

### Stage 1: RAW IMMUTABLE STORAGE

**Input:** Downloaded/existing raw files  
**Output:** Immutable raw data directory

- Raw data in `datasets/raw/{source_name}/` is **never modified** by the pipeline
- Each source directory must contain a `SOURCE_URL.txt`
- SHA-256 checksum of raw files recorded in source manifest

### Stage 2: SOURCE MANIFEST

**Input:** Raw data directories  
**Output:** `meditriage/data/build_temp/source_manifest.json`

```json
{
  "sources": {
    "mtsamples": {
      "path": "datasets/raw/mtsamples/",
      "files": ["mtsamples (1).csv"],
      "checksums": {"mtsamples (1).csv": "<SHA-256>"},
      "license": "CC0-1.0",
      "license_grade": "A",
      "source_url": "https://huggingface.co/datasets/NickyNicky/medical_mtsamples",
      "adapter_version": "1.1.0"
    }
  },
  "build_timestamp": "<ISO-8601>",
  "build_version": "v2.0.0",
  "pipeline_git_commit": "<SHA>"
}
```

### Stage 3: ADAPTER (Per-Source Ingestion)

**Input:** Raw data files  
**Output:** Standardized DataFrames per source (Parquet shards in `build_temp/01_ingest/`)

- Each adapter reads its raw format (CSV, JSON, JSONL, Parquet, fixed-width)
- Adapters emit DataFrames with the initial columns: `source_dataset`, `source_record_id`, `raw_text`, `language`, `department`, `triage_level`
- **NEISS downsampling** applied at this stage: stratified sample of 500,000 rows (from 7.1M), seed=42
- Adapters must handle missing/malformed data gracefully (skip, not crash)

### Stage 4: NORMALIZATION (Schema Alignment)

**Input:** Stage 3 ingested shards  
**Output:** Schema-aligned DataFrames (Parquet shards in `build_temp/02_normalize/`)

- Ensure all 26 canonical columns exist (populate with `NULL` if source lacks them)
- Set `text = raw_text` for all SOURCE records
- Set `provenance = "SOURCE"` for all records from this stage
- Set `augmentation_type = NULL`, `augmentation_parent_id = NULL`
- Generate `sample_id = "{source_dataset}::{source_record_id}::0"`
- Set `is_code_mixed = False` (default; may be updated by language classification)
- Set `dataset_version = "v2.0.0"`
- Set `license`, `license_url`, `source_url` per source from LICENSE REGISTER

### Stage 5: LABEL MAPPING

**Input:** Stage 4 normalized shards  
**Output:** Label-mapped DataFrames (Parquet shards in `build_temp/03_labels/`)

- **Department mapping:**
  - MTSamples: via `src/specialty_mapping.py` → `department_source = "mapped"`, `department_confidence = "high"`
  - NEISS: via adapter heuristic (diagnosis code + body part + narrative) → `department_source = "inferred"`, `department_confidence = "low"`
  - NHAMCS: all ED → `department_source = "native"`, `department_confidence = "high"`
  - Other sources: via adapter keyword classification → `department_source = "inferred"`

- **Severity mapping:**
  - NHAMCS: IMMEDR field → ESI 1–5 → `severity_source = "native_esi"`
  - Kaggle Medical Triage: string-to-ESI mapping → `severity_source = "mapped"`
    - `Emergency` → `S2`, `Urgent` → `S3`, `Routine` → `S4`, `Observation` → `S4`
  - All other sources: `triage_level = NULL`, `severity_source = "none"`

- **Severity labels are NEVER regex-heuristic-generated** in the canonical build (unlike the historical `normalize.py:score_severity()`)

### Stage 6: LANGUAGE CLASSIFICATION

**Input:** Stage 5 label-mapped shards  
**Output:** Language-classified DataFrames (Parquet shards in `build_temp/04_language/`)

- **Script detection:** count characters in Unicode ranges:
  - Latin: `[\u0000-\u007F\u0080-\u00FF\u0100-\u024F]`
  - Devanagari: `[\u0900-\u097F]`
  - CJK: `[\u4E00-\u9FFF\u3400-\u4DBF\uF900-\uFAFF]`
- **Language confidence:** `native` (from source metadata), `detected` (from script analysis)
- **Code-mixing detection:** if both Latin and Devanagari tokens exceed 10% of text → `is_code_mixed = True`
- **CJK safety filter:** any record with >5% CJK characters is flagged with `quality_flags = "cjk_detected"` and REJECTED from the canonical build

### Stage 7: PROVENANCE

**Input:** Stage 6 language-classified shards  
**Output:** Provenance-tagged DataFrames (Parquet shards in `build_temp/05_provenance/`)

- Set `provenance = "SOURCE"` for all records from stages 3–6
- Set `department_source` per the mapping in Stage 5
- Set `severity_source` per the mapping in Stage 5
- No augmentation-derived records exist at this stage

### Stage 8: QUALITY CONTROL

**Input:** Stage 7 provenance-tagged shards  
**Output:** Quality-filtered DataFrames (Parquet shards in `build_temp/06_quality/`)

- **Reject** records where `text` is empty, NULL, or whitespace-only
- **Reject** records where `text` length < 10 characters
- **Flag** records where `text` length > 2000 characters: `quality_flags += "|long_text"`
- **Flag** records where `text` length < 50 characters: `quality_flags += "|short_text"`
- **Reject** records with CJK content (per Stage 6 safety filter)
- **Flag** records with low-confidence department assignments
- Generate quality statistics

### Stage 9: DEDUPLICATION

**Input:** Stage 8 quality-filtered shards  
**Output:** Deduplicated DataFrames (Parquet shards in `build_temp/07_dedup/`)

- **Exact deduplication:** normalize text (strip whitespace, lowercase) → hash → drop duplicates
- **Priority-based retention:** when duplicates span sources, keep the record from the higher-priority source (priority: MTSamples > NHAMCS > NEISS > PMC-Patients > others)
- Record dropped duplicate IDs for audit

### Stage 10: LEAKAGE CONTROL

**Input:** Stage 9 deduplicated shards  
**Output:** Leakage-safe DataFrames (Parquet shards in `build_temp/08_leakage/`)

- Verify no `source_record_id` appears in more than one split
- Verify no near-duplicate texts cross split boundaries (fuzzy matching at >0.95 similarity on a sample)
- Log any detected leakage as a build failure

### Stage 11: SPLIT

**Input:** Stage 10 leakage-safe shards  
**Output:** Split-assigned DataFrames (Parquet shards in `build_temp/09_split/`)

- Split assignment via deterministic SHA-256 hash of `source_record_id` (NOT `sample_id`)
  - This ensures all augmented variants of a record land in the same split
- Split ratios: `train: 0.8`, `val: 0.1`, `test: 0.1`
- **Splits are assigned BEFORE augmentation** (critical rule)
- Verify split proportions are within ±1% of targets

### Stage 12: APPROVED AUGMENTATION

**Input:** Stage 11 split-assigned shards  
**Output:** Augmented DataFrames (Parquet shards in `build_temp/10_augment/`)

- **Only augment TRAINING split records** — validation and test sets receive NO augmentation
- Augmentation types:
  1. **Hinglish expansion** (provenance = A): deterministic phonetic transliteration
  2. **Clinical linguistic variation** (provenance = A): 10 variation styles from `meditriage/multilingual/variation/`
  3. **Hard-negative generation** (provenance = B): controlled adversarial pairs from `meditriage/multilingual/hard_negative/`
- Each augmented record must set:
  - `provenance = "A"` or `"B"` (never `"C"` without explicit approval)
  - `augmentation_type = "<specific_type>"`
  - `augmentation_parent_id = "<parent_sample_id>"`
  - `source_record_id = "<parent_source_record_id>"` (same as parent)
  - `split = "<parent_split>"` (same as parent)

### Stage 13: ROBUSTNESS STRATA

**Input:** Stage 12 augmented shards  
**Output:** Robustness-stratified DataFrames (Parquet shards in `build_temp/11_robustness/`)

- Tag records used for robustness evaluation with `robustness_stratum`
- Strata correspond to the 20 dimensions in the Multilingual Robustness Matrix (SPEC Section 6)

### Stage 14: SAFETY STRATA

**Input:** Stage 13 robustness-stratified shards  
**Output:** Safety-stratified DataFrames (Parquet shards in `build_temp/12_safety/`)

- Tag records used for safety evaluation with `red_flag_label` and `ood_stratum`
- OOD records (MedQA-USMLE, Medical Meadow) tagged with appropriate `ood_stratum`
- Red-flag evaluation records tagged per the strata in SPEC-07 Section 10

### Stage 15: FINAL VALIDATION

**Input:** Stage 14 safety-stratified shards  
**Output:** Validated canonical dataset

- Run full schema validation (all 26 columns present, correct types, enum compliance)
- Verify non-nullable fields have no NULLs
- Verify referential integrity (augmentation_parent_id references valid sample_id)
- Verify split isolation (no source_record_id crosses splits)
- Verify provenance completeness (every row has provenance ≠ NULL)
- Generate validation report

### Stage 16: MANIFEST + SHA-256

**Input:** Stage 15 validated dataset  
**Output:** Final canonical artifacts

- Export to `meditriage/data/canonical/v2.0.0/dataset.parquet` (single file or partitioned)
- Compute SHA-256 of final Parquet file
- Generate `build_manifest.json` with:
  - Source manifest reference
  - Pipeline git commit
  - Stage statistics (rows in/out per stage)
  - Split distributions
  - Department distributions
  - Severity distributions
  - Language distributions
  - Provenance distributions
  - Augmentation statistics
  - Quality filter statistics
  - Deduplication statistics
  - Total build time

### Stage 17: DATASET-GATE-01

**Input:** Stage 16 manifest + dataset  
**Output:** `docs/specification/audits/GATE_2_DATASET_GATE_REPORT.md`

- Evaluate all 18 DATASET-GATE-01 requirements
- Generate the required audit output format (per SPEC Section 5)
- Record `PASS` or `FAIL` with specific item numbers

---

## Split Strategy

### Rule: Split Before Augmentation

```
Source records → Assign split via hash(source_record_id)
                 ↓
              train (80%)
              val (10%)
              test (10%)
                 ↓
         Only train-split records → Augmentation pipeline
         Val/test records pass through UNAUGMENTED
```

### Leakage Prevention

1. **Same source record:** split assigned by `source_record_id`, not `sample_id` — all variants of the same record share the same split.
2. **Same patient/group:** where source data includes patient/group identifiers, group-based splitting is preferred.
3. **Same normalized text:** exact deduplication occurs before split assignment.
4. **Same augmentation family:** augmented records inherit their parent's split.

---

## Reproducibility Contract

The pipeline is fully deterministic given:

1. Fixed raw data (checksummed in source manifest)
2. Fixed adapter versions (recorded in source manifest)
3. Fixed random seed (42)
4. Fixed pipeline git commit (recorded in build manifest)
5. Fixed split ratios (0.8/0.1/0.1)
6. Fixed downsampling parameters (NEISS: 500K, seed=42)
7. Fixed augmentation configuration

Any change to the above requires a new dataset version (e.g., `v2.1.0`).

---

## Output Artifacts

| Artifact | Path | Description |
|---|---|---|
| Canonical Parquet | `meditriage/data/canonical/v2.0.0/dataset.parquet` | Final canonical dataset |
| Build Manifest | `meditriage/data/canonical/v2.0.0/build_manifest.json` | Full build metadata |
| Source Manifest | `meditriage/data/canonical/v2.0.0/source_manifest.json` | Raw data checksums and versions |
| Gate Report | `docs/specification/audits/GATE_2_DATASET_GATE_REPORT.md` | DATASET-GATE-01 evaluation |
| Quality Report | `meditriage/data/canonical/v2.0.0/quality_report.json` | Quality filter statistics |
| Dedup Report | `meditriage/data/canonical/v2.0.0/dedup_report.json` | Deduplication statistics |
| Split Report | `meditriage/data/canonical/v2.0.0/split_report.json` | Split distribution verification |

---

## Prohibited Operations

1. **No manual spreadsheet editing** of the dataset
2. **No manual label editing** — labels come from adapters or `NULL`
3. **No manual copy/paste construction** of training rows
4. **No regex-heuristic severity assignment** on unlabeled sources (use `NULL` instead)
5. **No augmentation of validation/test sets**
6. **No silent LLM-generated data incorporation** — must be tagged as provenance `C`
7. **No use of Grade D/E licensed data** in the canonical build
