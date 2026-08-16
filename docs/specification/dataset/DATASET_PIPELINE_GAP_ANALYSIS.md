# MediTriageAI — Dataset Pipeline Gap Analysis

**Specification Baseline:** v1.0.0-FROZEN  
**Gate:** GATE 2 — Canonical Dataset Reconstruction  
**Date:** 2026-08-16  
**Status:** AUTHORITATIVE

---

## Existing Infrastructure Audit

This document audits every component of the existing `meditriage/builder/` pipeline against the requirements of the canonical dataset build specification (`DATASET_BUILD_SPEC.md`).

---

## Adapters

### Status Summary

| Adapter | File | Status | Issues |
|---|---|---|---|
| MTSamples | `adapters/mtsamples.py` | **EXISTS_BUT_NEEDS_FIX** | Emits extra columns (`tracking_id`, `seed_id`, `text`, `routing_confidence`, `severity_label_source`, `is_perturbed`, `variant_index`, `extraction_timestamp`, `original_schema_version`). Imports `src.specialty_mapping` directly (cross-module dependency). Must align to canonical schema. |
| NEISS | `adapters/neiss.py` | **EXISTS_AND_VALID** | Works correctly. Emits `dataset_source`, `raw_text`, `department`, `triage_level`, `language`. Age<18→PEDS override is a design choice, not a bug. Downsampling must be added externally (not adapter's responsibility). |
| NHAMCS ED | `adapters/nhamcs_ed.py` | **EXISTS_AND_VALID** | Works correctly. Parses fixed-width CDC format. Emits ESI triage levels from IMMEDR field. |
| Symptom2Disease | `adapters/symptom2disease.py` | **EXISTS_BUT_NEEDS_FIX** | Emits extra columns (`tracking_id`, `seed_id`). Must align to canonical schema. |
| MedQA-USMLE | `adapters/medqa_usmle.py` | **EXISTS_AND_VALID** | Works correctly for Q&A ingestion. Role change: REFERENCE_ONLY (not primary training). |
| Medical Meadow MedQA | `adapters/medical_meadow_medqa.py` | **EXISTS_AND_VALID** | Works correctly. Role change: REFERENCE_ONLY. |
| FedMML ED Triage | `adapters/fedmml_ed_triage.py` | **EXISTS — QUARANTINED** | Adapter works but source dataset is QUARANTINED (License Grade D). Adapter should remain but NOT be included in active builds until license is cleared. |
| ChatDoctor HealthCareMagic | `adapters/chatdoctor_healthcaremagic.py` | **OBSOLETE** | Source REJECTED (License Grade E). Adapter should be preserved but EXCLUDED from active builds. |
| ChatDoctor iCliniq | `adapters/chatdoctor_icliniq.py` | **OBSOLETE** | Source REJECTED (License Grade E). Same as above. |
| PMC-Patients | `adapters/pmc_patients.py` | **EXISTS_AND_VALID** | Works correctly. License: CC BY-NC-SA 4.0. Must carry license metadata in canonical schema. |
| Kaggle Medical Triage | `adapters/kaggle_medical_triage.py` | **EXISTS_BUT_NEEDS_FIX** | Severity labels are string-based (Routine/Urgent/Emergency/Observation); must add ESI mapping logic. |
| L3Cube Code-Mixed | `adapters/l3cube_code_mixed.py` | **EXISTS_AND_VALID** | Works correctly. Hinglish linguistic patterns. Role: AUXILIARY_LANGUAGE. |
| MedDialog EN | `adapters/meddialog_en.py` | **EXISTS — QUARANTINED** | Adapter has v2.0 CJK safety filter (good), but source data is 100% CJK. Source QUARANTINED (License Grade D + mislabeled content). |

### Adapter Gap Summary

| Gap | Affected Adapters | Resolution |
|---|---|---|
| Non-canonical schema output | MTSamples, Symptom2Disease | Update adapters to emit only the 6 core ingestion columns: `source_dataset`, `source_record_id`, `raw_text`, `language`, `department`, `triage_level` |
| Cross-module dependency | MTSamples (`from src.specialty_mapping`) | Copy `RAW_TO_DEPARTMENT` mapping into adapter or create shared mapping module in `meditriage/builder/` |
| Missing ESI severity mapping | Kaggle Medical Triage | Add string-to-ESI mapping in adapter |
| Missing downsampling | NEISS | Add downsampling logic in orchestrator (not adapter) |
| Missing `source_record_id` | All adapters | Adapters should emit `source_record_id` as a distinct field (currently some use `tracking_id`, some use nothing) |

---

## Pipeline Stages (Orchestrator)

### Current Orchestrator (`meditriage/builder/orchestrator.py`)

| Stage | Canonical Requirement | Current Status | Gap |
|---|---|---|---|
| Source Manifest | Stage 2: Record raw checksums, versions | **MISSING** | No source manifest generation. Build manifest only records adapter versions and timing. |
| Ingest | Stage 3: Per-source adapter | **EXISTS_AND_VALID** | Works correctly via `ADAPTER_REGISTRY`. |
| Normalize | Stage 4: Schema alignment | **EXISTS_BUT_NEEDS_FIX** | Current Stage 2 is a no-op (copies parquet files unchanged). Must implement canonical schema alignment. |
| Label Mapping | Stage 5: Department + severity | **MISSING** | Department mapping happens inside adapters (inconsistently). Severity mapping is absent (historical `normalize.py:score_severity()` is regex-heuristic and NOT used by the orchestrator). |
| Language Classification | Stage 6: Script detection | **MISSING** | No script-level language classification exists in the orchestrator. |
| Provenance | Stage 7: Text + label provenance | **MISSING** | No provenance tagging. The `provenance` field does not exist in the historical schema. |
| Quality Control | Stage 8: Filtering + flagging | **MISSING** | No quality control stage. Current Stage 3 only adds NULL columns if missing. |
| Deduplication | Stage 9: Exact + priority dedup | **EXISTS_AND_VALID** | Current Stage 4 implements priority-based exact deduplication. Works correctly. |
| Leakage Control | Stage 10: Cross-split leakage | **MISSING** | Leakage detection exists in `stages/validate.py` but is NOT called by the orchestrator. |
| Split | Stage 11: Deterministic hash | **EXISTS_AND_VALID** | Current Stage 6 implements MD5-based split. Must migrate to SHA-256 for consistency with spec. Must use `source_record_id` not `id`. |
| Augmentation | Stage 12: Multilingual + variation | **EXISTS_BUT_BROKEN** | The augmentation stage (Stage 5) is a **file-copy no-op** (lines 171-174). The actual `apply_augmentation()` function exists in `stages/augment.py` and works, but is NEVER CALLED. |
| Robustness Strata | Stage 13 | **MISSING** | No robustness stratum tagging. |
| Safety Strata | Stage 14 | **MISSING** | No safety/OOD stratum tagging. |
| Final Validation | Stage 15 | **MISSING** | No final schema validation in the orchestrator. `schema.py:validate_schema()` exists but is NOT called. |
| Manifest + SHA-256 | Stage 16 | **EXISTS_BUT_NEEDS_FIX** | Build manifest exists but lacks: source manifest reference, git commit, per-stage statistics, provenance distributions, quality statistics. No SHA-256 of final dataset. |
| DATASET-GATE-01 | Stage 17 | **MISSING** | No gate evaluation. |

---

## Standalone Stage Modules (`meditriage/builder/stages/`)

| Module | Status | Notes |
|---|---|---|
| `augment.py` | **EXISTS_AND_VALID** | Correctly calls `MultilingualTranslator.expand_dataframe()`. Problem: orchestrator never calls it. |
| `deduplicate.py` | **EXISTS_BUT_NEEDS_FIX** | Works on `text` and `seed_id` columns — must be adapted to canonical schema (`raw_text`, `source_record_id`). |
| `normalize.py` | **EXISTS_BUT_NEEDS_FIX** | Contains `score_severity()` regex heuristic — this MUST NOT be used in canonical build (per build spec: "No regex-heuristic severity assignment on unlabeled sources"). Department mapping is a subset of what adapters already do. |
| `split.py` | **EXISTS_AND_VALID** | SHA-256-based split assignment. Uses `seed_id` — must adapt to `source_record_id`. |
| `validate.py` | **EXISTS_AND_VALID** | Schema validation + leakage detection. Not called by orchestrator. Must be called. |

---

## Multilingual Infrastructure

| Component | Path | Status | Notes |
|---|---|---|---|
| **MultilingualTranslator** | `multilingual/translator.py` | **EXISTS_AND_VALID** | Full orchestration: provider selection, caching, validation, variation engine integration. |
| **MultilingualConfig** | `multilingual/config.py` | **EXISTS_AND_VALID** | Supports target languages, provider selection, quality thresholds. |
| **ClinicalQualityValidator** | `multilingual/validator.py` | **EXISTS_AND_VALID** | Validates clinical quality of translations. |
| **MultilingualCache** | `multilingual/cache.py` | **EXISTS_AND_VALID** | Persistent caching for translations. |
| **Providers** | `multilingual/providers/` | **EXISTS_AND_VALID** | Three providers: `offline` (deterministic), `gemini`, `openai`. |
| **Offline Provider** | `multilingual/providers/offline.py` | **EXISTS_AND_VALID** | Deterministic rule-based Hinglish generation. Provenance = A. |
| **Variation Engine** | `multilingual/variation/engine.py` | **EXISTS_AND_VALID** | Orchestrates 10 clinical variation generators. |
| **Variation Generators** | `multilingual/variation/generators.py` | **EXISTS_AND_VALID** | 10 style generators (lexical, syntactic, conversational, ED, physician, nurse, patient, abbreviated, formal, colloquial Indian). |
| **Variation Config** | `multilingual/variation/config.py` | **EXISTS_AND_VALID** | Configuration for variation budget, styles. |
| **Variation Validator** | `multilingual/variation/validator.py` | **EXISTS_AND_VALID** | Clinical semantic preservation validation. |
| **Hard Negative Engine** | `multilingual/hard_negative/` | **EXISTS_AND_VALID** | Hard-negative generation with library, validator, config. |
| **Phenotype Engine** | `multilingual/phenotype/` | **EXISTS_AND_VALID** | Phenotype-based augmentation with clinical rules. |
| **Report Generator** | `multilingual/report.py` | **EXISTS_AND_VALID** | Multilingual expansion report generation. |

---

## Existing Quality/Validation Infrastructure

| Component | Status | Notes |
|---|---|---|
| `builder/schema.py:validate_schema()` | **EXISTS_BUT_NEEDS_FIX** | Validates against 15-column extended schema (not the canonical 26-column schema). Must be updated. |
| `builder/schema.py:REQUIRED_COLUMNS` | **EXISTS_BUT_NEEDS_FIX** | 15 columns defined; must be extended to 26 canonical columns. |
| `stages/validate.py:validate_dataframe()` | **EXISTS_AND_VALID** | Calls validate_schema + leakage check. Must be called by orchestrator. |
| Dedup report | **EXISTS_AND_VALID** | `orchestrator.py` writes `duplicate_report.txt`. |
| Coverage report | **EXISTS_AND_VALID** | `orchestrator.py` writes `coverage_report.txt`. |
| Dataset statistics | **EXISTS_AND_VALID** | `orchestrator.py` writes `dataset_statistics.json`. |

---

## Critical Gaps Summary

| Priority | Gap | Impact | Resolution |
|---|---|---|---|
| **P0 — CRITICAL** | Stage 5 augmentation is a no-op | Zero multilingual/robustness augmentation in any build | Wire `apply_augmentation()` call into orchestrator |
| **P0 — CRITICAL** | No provenance tracking | Cannot satisfy FR-DATA-01, FR-TEXT-01, DATASET-GATE-01 item 14 | Add provenance columns to schema and populate throughout pipeline |
| **P0 — CRITICAL** | No source manifest with checksums | Cannot satisfy DATASET-GATE-01 item 1 | Implement source manifest generation |
| **P0 — CRITICAL** | No DATASET-GATE-01 evaluation | Cannot pass Gate 2 | Implement gate evaluation stage |
| **P0 — CRITICAL** | No SHA-256 of final dataset | Cannot satisfy DATASET-GATE-01 item 16 | Add checksum computation to export |
| **P1 — HIGH** | No language/script classification | Cannot detect CJK contamination or verify multilingual composition | Implement script detection stage |
| **P1 — HIGH** | No quality control filtering | Garbage-in-garbage-out risk | Implement quality control stage |
| **P1 — HIGH** | Schema mismatch (7 cols → 26 cols) | Cannot store provenance, license, quality flags | Update schema definition and all emitters |
| **P1 — HIGH** | ChatDoctor adapters still in ADAPTER_REGISTRY | Rejected sources could be accidentally included | Add license-gate enforcement to orchestrator |
| **P2 — MEDIUM** | Leakage validation not called | Cannot satisfy DATASET-GATE-01 items 10, 13 | Wire validate_dataframe into orchestrator after split |
| **P2 — MEDIUM** | Split uses MD5, not SHA-256 | Inconsistent with spec (minor, same behavior) | Migrate to SHA-256 |
| **P2 — MEDIUM** | NEISS downsampling not implemented | 70% dominance if uncapped | Add stratified downsampling |
| **P3 — LOW** | MTSamples imports from `src/` | Cross-module dependency | Inline the mapping or create shared module |

---

## Components That Do NOT Need Changes

| Component | Rationale |
|---|---|
| `adapters/neiss.py` | Works correctly, emits correct columns |
| `adapters/nhamcs_ed.py` | Works correctly, provides critical ESI labels |
| `adapters/pmc_patients.py` | Works correctly |
| `adapters/l3cube_code_mixed.py` | Works correctly for Hinglish patterns |
| `multilingual/translator.py` | Full multilingual expansion orchestration |
| `multilingual/providers/offline.py` | Deterministic Hinglish generation |
| `multilingual/variation/generators.py` | 10 clinical variation generators |
| `multilingual/hard_negative/` | Complete hard-negative pipeline |
| `multilingual/phenotype/` | Complete phenotype augmentation |
| `multilingual/validator.py` | Clinical quality validation |
| `multilingual/cache.py` | Translation caching |
| `stages/split.py` | Hash-based deterministic split (minor SHA-256 migration) |
