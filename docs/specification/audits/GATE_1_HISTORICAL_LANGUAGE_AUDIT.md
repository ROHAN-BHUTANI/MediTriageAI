# Gate 1 — Historical Language-Distribution Audit

**Specification Baseline:** MediTriageAI v1.0.0-FROZEN (`docs/specification/frozen/v1.0.0/`)  
**Audit Date:** 2026-08-15  
**Auditor:** Antigravity Senior Repository Analyst & Governance Agent  
**Status:** **COMPLETE**  
**Classification:** GOVERNANCE AUDIT ONLY — No source code, weights, or datasets modified.

---

## 1. Audit Objective

The objective of Gate 1 is to determine empirically what dataset and training artifacts were actually used in historical MediTriageAI training run(s), analyze the real composition of that dataset (language distribution, script presence, provenance, specialist routing, severity distribution, duplication, and multilingual robustness), evaluate it against the 18 requirements of `DATASET-GATE-01`, and classify its eligibility for the final benchmark campaign.

---

## 2. Repository Evidence Examined

The following empirical sources within the repository were audited directly:

1. **Model Checkpoint Artifacts:**
   - `results/xlm_roberta_large/checkpoint.pt` (1,112,416,777 bytes, SHA-256: `4479b86db7eb946209e19a5cedb30adeb3863f694dc5b19a51840479d0ddf7fc`)
   - `results/xlm_roberta_large/metadata.json` (Commit: `3760b26e57599c4faadb9728f7470b19c1cc74e5`, Timestamp: `2026-08-05T15:35:05Z`)
   - `results/xlm_roberta_large/metrics.json` (`n_test_rows`: 800, `specialist_macro_f1`: 0.0066, `severity_macro_f1`: 0.0190)
   - `results/emergent_path_triage/checkpoint.pt` (1,116,635,881 bytes, SHA-256: `b9cb29469b606c4c9a2a092042ff1004bbe8498cc33e282cfbff02277a6890ae`)
   - `results/emergent_path_triage/metrics.json` (`n_test_rows`: 800, `specialist_macro_f1`: 0.0454, `severity_macro_f1`: 0.0031)

2. **Dataset Artifacts on Disk:**
   - `meditriage/data/processed/dataset.parquet` (1,306,576,608 bytes, SHA-256: `f36c2ae25315c43036dd80e24557dc4852d024bddaaca82bcd4bd9bcfbc149c8`)
   - `meditriage/data/processed/dataset.csv` (7,850,024,739 bytes)
   - `meditriage/data/processed/build_manifest.json` (Build timestamp: `1785595049.34`, 13 adapters)
   - `meditriage/data/processed/dataset_statistics.json` (Total rows: 10,230,264)
   - `data/clinical_triage_clean.csv` (14,190,029 bytes, SHA-256: `894bccbcf3e4eb61a1490e32d4d6729ae1eefba8f1dfc6eae1305ccee909e9fd`)
   - `data/clinical_triage_hinglish.csv` (1,744,984 bytes, SHA-256: `26c1d95fd5bcca50bb6ed4420c99fef7f596050a8f769c5c3814c0acc68bd84f`)

3. **Pipeline Source Code:**
   - `scripts/train.py` (Default dataset pointer: `DEFAULT_DATASET = REPO_ROOT / "meditriage" / "data" / "processed" / "dataset.parquet"`)
   - `scripts/train_ddp.py` (Loads `meditriage/data/processed/dataset.parquet`, fallback to `.csv`)
   - `src/dataset.py` (Schema loader `load_split_rows`)
   - `meditriage/builder/orchestrator.py` (Stage 5 augmentation execution audit)

4. **Historical Forensic Records:**
   - `results/multilingual_forensic/00_EXECUTIVE_FORENSIC_REPORT.md`
   - `results/multilingual_forensic/02_language_forensic.md`
   - `results/multilingual_forensic/05_training_lineage.md`
   - `results/multilingual_forensic/11_checkpoint_inventory.json`
   - `results/multilingual_forensic/13_dataset_identity_reconciliation.md`

---

## 3. Relevant Training Runs

| Run Identifier | Timestamp | Model / Architecture | Dataset Target | Execution Environment | Status / Artifact Quality |
|---|---|---|---|---|---|
| **Run A (Most Recent Completed Run)** | 2026-08-05T15:35:05Z | `XLMRobertaLargeModel` (`xlm-roberta-base` weights, 278M params) | `meditriage/data/processed/dataset.parquet` | CPU-only (Windows clone) | **Smoke-test checkpoint only** (epoch=0, step=0, 800 test rows, near-random F1: 0.0066 / 0.0190) |
| **Run B (Experimental Architecture Run)** | 2026-08-05T15:25:17Z | `EmergentPathTriageModel` (E-PATH, `xlm-roberta-base` backbone) | `meditriage/data/processed/dataset.parquet` | CPU-only (Windows clone) | **Smoke-test checkpoint only** (epoch=0, step=0, 800 test rows, F1: 0.0454 / 0.0031) |
| **Run C (DGX Production Run Referenced)** | Prior to 2026-08-05 | `best_model.pt` (referenced in historical logs) | `dataset.parquet` (10.2M rows) | DGX Multi-GPU | **ABSENT from local repository clone** (SHA-256 `9b4a15e1467747aa53ac6bc37fd5ae766f70de50ef79af81f44fad5d569fa687` recorded in reports) |

**Conclusion [FACT]:** The local repository contains only smoke-test checkpoints (Runs A and B). The production DGX checkpoint (Run C) is not located on this clone. All local training configurations target `meditriage/data/processed/dataset.parquet`.

---

## 4. Actual Dataset Artifact

- **Canonical Path:** `meditriage/data/processed/dataset.parquet`
- **Secondary Format:** `meditriage/data/processed/dataset.csv`
- **Total Ingested Rows:** 10,230,264
- **Schema Columns [FACT]:** `id` (string), `split` (string), `dataset_source` (string), `language` (string), `raw_text` (string), `department` (string), `triage_level` (string)
- **Split Distribution [MEASURED]:**
  - `train`: 8,183,157 rows (79.99%)
  - `val`: 1,024,105 rows (10.01%)
  - `test`: 1,023,002 rows (10.00%)

---

## 5. Dataset SHA-256

- **On-Disk Checksum [MEASURED]:**  
  `f36c2ae25315c43036dd80e24557dc4852d024bddaaca82bcd4bd9bcfbc149c8` (Parquet)  
- **Historical Pre-Export Checksum [FACT from reconciliation log]:**  
  `bc9160fcb8e6e4413e7a4e06d2dcab5e0e1c84b17f801f9eb364acb82192f9fb`  
- **Classification:** **MATCH with documented Parquet re-export** (as reconciled in `results/multilingual_forensic/13_dataset_identity_reconciliation.md`; row count and tabular records are identical across all 10,230,264 rows).

---

## 6. Dataset Size

- **Total Rows:** 10,230,264
- **Parquet File Size:** 1,306,576,608 bytes (1.22 GB)
- **CSV File Size:** 7,850,024,739 bytes (7.31 GB)
- **Source Breakdown [MEASURED]:**

| Source Dataset | Count | Percentage |
|---|---|---|
| `neiss` | 7,137,339 | 69.77% |
| `meddialog_en` | 2,616,894 | 25.58% |
| `pmc_patients` | 167,034 | 1.63% |
| `chatdoctor_healthcaremagic` | 112,002 | 1.09% |
| `fedmml_ed_triage` | 84,177 | 0.82% |
| `nhamcs_ed` | 41,509 | 0.41% |
| `l3cube_code_mixed` | 37,725 | 0.37% |
| `medqa_usmle` | 11,449 | 0.11% |
| `medical_meadow_medqa` | 10,178 | 0.10% |
| `chatdoctor_icliniq` | 7,321 | 0.07% |
| `mtsamples` | 2,371 | 0.02% |
| `symptom2disease` | 1,153 | 0.01% |
| `kaggle_medical_triage` | 1,112 | 0.01% |
| **Total** | **10,230,264** | **100.00%** |

---

## 7. Language Distribution

### A. Dataset Column Label Distribution [MEASURED — 100% of rows]

| Language Label | Row Count | Percentage |
|---|---|---|
| `en` (English) | 10,192,539 | **99.6312%** |
| `hi-en` (Hinglish) | 37,725 | **0.3688%** |
| `hi` (Hindi Devanagari) | 0 | **0.0000%** |
| `hi-Latn` (Roman Hindi) | 0 | **0.0000%** |
| `en-hi` (Code-switched) | 0 | **0.0000%** |

### B. Independent Script & Token Detection [MEASURED — 204,437 rows sampled across all 152 row groups, 1 in 50 rows]

| Linguistic Feature | Regex / Criteria | Sample Count | Sample % | Estimated Total Rows |
|---|---|---|---|---|
| **Hindi Devanagari Script** | `[\u0900-\u097F]` | **0** | **0.0000%** | **0** |
| **CJK Script (Chinese Contamination)** | `[\u4E00-\u9FFF\u3400-\u4DBF\uF900-\uFAFF]` | **52,351** | **25.6074%** | **~2,619,000** |
| **Arabic Script** | `[\u0600-\u06FF]` | 0 | 0.0000% | 0 |
| **Cyrillic Script** | `[\u0400-\u04FF]` | 3 | 0.0015% | ~150 |
| **Hinglish Token Markers** | Common Romanized Hindi function words | 35,052 | 17.1456% | — |

### C. Language Distribution Findings [FACT]

1. **Severe English Dominance:** The dataset is **99.63% English** by metadata label.
2. **Zero Hindi Devanagari:** Exactly **0** Devanagari characters were found across the 204,437 sampled rows. Standard Hindi is completely absent.
3. **Zero Generated Multilingual Data:** The 37,725 `hi-en` rows originate entirely from a single source (`l3cube_code_mixed`). The multilingual augmentation pipeline (`MultilingualTranslator`, variation engine, phenotype expansion) was bypassed in the production orchestrator (`orchestrator.py:171-174` was a file-copy no-op).
4. **Massive CJK Contamination:** **25.61% of sampled rows** (originating almost entirely from `meddialog_en`, 2.6M rows) contain Chinese text that was ingested without script filtering and mislabeled as `language=en`.

---

## 8. Provenance Distribution

| Provenance Category (per SPEC-04 / ADR-009) | Status in Parquet | Source Contribution | Count | Percentage |
|---|---|---|---|---|
| `SOURCE` (Unmodified real clinical text) | Inferred | NEISS, MedDialog, PMC-Patients, ChatDoctor, NHAMCS, MTSamples, Symptom2Disease | 10,097,185 | 98.70% |
| `A` (Deterministic linguistic augmentation) | **ABSENT** | Bypassed in orchestrator | 0 | 0.00% |
| `B` (Rule/template construction) | **ABSENT** | Bypassed in orchestrator | 0 | 0.00% |
| `C` (LLM-generated synthetic data) | Inferred | `fedmml_ed_triage` | 84,177 | 0.82% |
| Excluded sources present in dataset | Inferred | `l3cube_code_mixed` (37,725), `medqa_usmle` (11,449), `medical_meadow_medqa` (10,178) | 59,352 | 0.58% |

**Key Finding [FACT]:** The `label_provenance` and `text_provenance` fields are **MISSING** from `meditriage/data/processed/dataset.parquet`. Provenance cannot be traced per-sample within the dataset columns.

---

## 9. Specialist Distribution

| Department Label | Row Count | Percentage of Total | Supervision Status |
|---|---|---|---|
| `GEN_MED` | 3,159,934 | 30.89% | Canonical |
| `PEDS` | 2,691,856 | 26.31% | Canonical |
| `ORTHO` | 1,813,489 | 17.73% | Canonical |
| `ENT_OPHTHALMO` | 1,213,464 | 11.86% | Canonical |
| `NEURO` | 638,147 | 6.24% | Canonical |
| `CARDIO_PULM` | 342,508 | 3.35% | Canonical |
| `ED` | 125,703 | 1.23% | Canonical |
| `RENAL_URO` | 66,650 | 0.65% | Canonical |
| `SURGERY` | 62,332 | 0.61% | Canonical |
| `ONCOLOGY_HEME` | 56,709 | 0.55% | Canonical |
| `GI` | 27,491 | 0.27% | Canonical |
| `OBGYN` | 17,765 | 0.17% | Canonical |
| `PSYCH` | 2,926 | 0.03% | Canonical |
| Unmapped string labels (Cardiology, Dermatology, etc.) | 1,092 | 0.01% | Unmapped (from Kaggle / Symptom2Disease) |
| Missing / None (`medical_meadow_medqa`) | 10,178 | 0.10% | Unsupervised |
| **Total with Department Annotation** | **10,220,086** | **99.90%** | — |

**Key Finding [MEASURED]:** The top 4 departments (`GEN_MED`, `PEDS`, `ORTHO`, `ENT_OPHTHALMO`) account for **86.79%** of the entire dataset. Low-volume classes like `PSYCH` (0.03%) and `OBGYN` (0.17%) have severe class imbalance (imbalance ratio > 1,000:1).

---

## 10. Severity Distribution

| Severity / Triage Level | Row Count | % of Total Dataset | % of Annotated Severity | Source Concentration |
|---|---|---|---|---|
| **None (Missing Severity)** | **10,116,554** | **98.8885%** | — | NEISS, MedDialog, PMC-Patients, ChatDoctor, MTSamples (0% coverage) |
| `3` (Moderate) | 54,311 | 0.5309% | 47.76% | `fedmml_ed_triage` + `nhamcs_ed` |
| `4` (Low / Less Urgent) | 30,281 | 0.2960% | 26.63% | `fedmml_ed_triage` + `nhamcs_ed` |
| `2` (Emergent) | 20,665 | 0.2020% | 18.17% | `fedmml_ed_triage` + `nhamcs_ed` |
| `5` (Non-urgent) | 5,976 | 0.0584% | 5.26% | `fedmml_ed_triage` + `nhamcs_ed` |
| `1` (Resuscitation) | 1,365 | 0.0133% | 1.20% | `fedmml_ed_triage` + `nhamcs_ed` |
| Unmapped string labels (Routine, Urgent, Emergency, Observation) | 1,112 | 0.0109% | 0.98% | `kaggle_medical_triage` |
| **Total Annotated Severity** | **113,710** | **1.1115%** | **100.00%** | — |

### Severity Source Concentration [FACT]:
- `fedmml_ed_triage` (Category C, LLM-generated): **84,177 rows (74.03% of all severity labels)**.
- `nhamcs_ed`: **28,421 rows (24.99% of all severity labels)**.
- `kaggle_medical_triage`: **1,112 rows (0.98% of all severity labels)**.
- **74% of the entire severity supervision signal is LLM-generated synthetic data.**

---

## 11. Duplicate / Leakage Indicators

- **Sample Duplication [MEASURED — 100K sample]:** 100,000 unique texts (0.00% exact duplicate rate in the subsample). Full dataset duplicate scan from builder audit showed 10,699 duplicated complaints across 10.2M rows (0.10%).
- **Split Leakage [MEASURED]:** Stratified splits maintain train (80%) / val (10%) / test (10%) row counts. However, patient-level identifiers are absent (`repeated_patients: "No patient_id column"`), meaning patient-level leakage cannot be audited or prevented in the current schema.
- **Contamination Leakage:** 2.6M Chinese texts from `meddialog_en` are distributed across all three splits (train: ~2.09M, val: ~262K, test: ~262K), causing non-English cross-split noise.

---

## 12. Multilingual Robustness Presence

| Dimension / Feature | Presence in Historical Dataset | Evidence / Notes |
|---|---|---|
| Standard English | **PRESENT** | 10,192,539 rows (99.63%) |
| Standard Hindi (Devanagari) | **ABSENT** | 0 rows, 0 script occurrences in 204K sample |
| Romanized Hindi (`hi-Latn`) | **ABSENT** | 0 rows generated or labeled |
| Hinglish / Code-mixed | **PRESENT** (Incidental) | 37,725 rows from `l3cube_code_mixed` (0.37%) |
| Phonetic Transliteration Variants | **ABSENT** | Orchestrator Stage 5 bypassed |
| Spelling & Informal Chat Variants | **ABSENT** | Orchestrator Stage 5 bypassed |
| Clinical Shorthand / Abbreviations | **UNKNOWN** | Incidental presence in raw NEISS/MTSamples text only |
| ASR-like Transcription Noise | **ABSENT** | Orchestrator Stage 5 bypassed |
| Hard-Negative Clinical Cases | **ABSENT** | Orchestrator Stage 5 bypassed |
| Out-Of-Distribution (OOD) Queries | **ABSENT** from training | Stored in separate file `data/ood_queries.csv` |

---

## 13. DATASET-GATE-01 Compliance Matrix

Evaluation of `meditriage/data/processed/dataset.parquet` against the 18 requirements of `DATASET-GATE-01`:

| # | Requirement | Status | Evidence / Reason |
|---|---|---|---|
| 1 | Raw source datasets versioned and checksummed | **PARTIAL** | `build_manifest.json` versions adapters, but raw source checksums are unverified |
| 2 | Canonical ingestion complete | **FAIL** | Contains unmapped department strings, unmapped severity strings, and un-excluded datasets (`l3cube_code_mixed`, `medical_meadow_medqa`, `medqa_usmle`) |
| 3 | Multilingual expansion complete | **FAIL** | Stage 5 bypassed; 0 rows expanded |
| 4 | Hinglish/romanization variation generation complete | **FAIL** | 0 generated rows; only raw unaugmented L3Cube |
| 5 | Linguistic robustness augmentation complete | **FAIL** | 0 rows generated |
| 6 | Phenotype augmentation complete | **FAIL** | 0 rows generated |
| 7 | Hard-negative generation complete | **FAIL** | 0 rows generated |
| 8 | Quality validation passes | **FAIL** | Massive CJK contamination (~2.6M rows, ~25.6% of dataset) mislabeled as English |
| 9 | Deduplication passes | **PASS** | Deduplication report recorded 0 exact duplicates in clean builder runs |
| 10 | Train/validation/test leakage audit passes | **UNKNOWN** | No patient ID column to verify clinical isolation |
| 11 | Language-distribution report generated & reviewed | **FAIL** | 99.63% English; Hindi completely absent; CJK mislabeled |
| 12 | Class-distribution report generated | **PASS** | Class distribution recorded and analyzed |
| 13 | Train/val/test isolation explicitly verified | **PARTIAL** | Index splits verified; clinical patient isolation UNKNOWN |
| 14 | Provenance recorded per sample (`SOURCE`/`A`/`B`/`C`) | **FAIL** | No provenance column in dataset |
| 15 | Synthetic-vs-source proportions reported by category | **FAIL** | Not tracked in dataset schema |
| 16 | Final dataset receives a SHA-256 checksum | **PASS** | SHA-256: `f36c2ae25315c43036dd80e24557dc4852d024bddaaca82bcd4bd9bcfbc149c8` |
| 17 | Training configuration references exact checksum | **FAIL** | Training configs reference file path, not checksum hash |
| 18 | DGX training run records dataset checksum | **FAIL** | Historical checkpoint metadata records `dataset_manifest_hash: "unknown"` |

**DATASET-GATE-01 Overall Result: FAIL** (10 of 18 items FAIL, 2 PARTIAL, 1 UNKNOWN, 3 PASS).

---

## 14. Historical Artifact Eligibility

### Eligibility Determination: **NOT ELIGIBLE**

Per the binding rule established in SPEC-05 / ADR-008 and Amendment 1 of the frozen specification:
> *"If an old training artifact fails DATASET-GATE-01, it is classified NOT ELIGIBLE for the final benchmark campaign — full stop, no undocumented rescue attempt, no partial-credit reasoning."*

**The historical dataset artifact (`meditriage/data/processed/dataset.parquet`) fails 10 mandatory requirements of DATASET-GATE-01 and is formally classified as NOT ELIGIBLE for the final DGX benchmark campaign.**

---

## 15. Unknowns / Limitations

1. **DGX Production Checkpoint (`best_model.pt`):** Not present on this local clone. Evaluated metrics in paper drafts cannot be cryptographically verified against local checkpoint weights until retrieved from DGX storage.
2. **Patient-Level Leakage:** Cannot be audited on historical data because source adapters did not extract or preserve patient IDs.
3. **CJK Filtering Failure:** Root cause of why `meddialog_en` ingestion adapter failed to filter out Chinese dialogues despite code intent is documented as an adapter regex deficiency.

---

## 16. Recommended Next Gate

**GATE 2 — Dataset Gate Implementation**

Per the post-freeze implementation sequence (SPEC Section 24 / Amendment 5), Gate 2 must implement the actual `DATASET-GATE-01` validation engine in code:
- Checksum verification mechanism and fail-loudly training entrypoint
- Gate report generator enforcing all 18 requirements
- Extended canonical schema incorporating `label_provenance` (`DIRECT`/`MAPPED`/`INFERRED`) and `text_provenance` (`SOURCE`/`A`/`B`/`C`)
- Adapter filtering for CJK contamination and strict exclusion of `l3cube_code_mixed`, `medical_meadow_medqa`, and `medqa_usmle` from active training runs without explicit override flags.

---
*Report certified by Antigravity Agent. Gate 1 is COMPLETE.*
