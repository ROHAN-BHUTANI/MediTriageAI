# MediTriageAI — Canonical Dataset Selection

**Specification Baseline:** v1.0.0-FROZEN  
**Gate:** GATE 2 — Canonical Dataset Reconstruction  
**Date:** 2026-08-16  
**Status:** AUTHORITATIVE  
**Governing Document:** `docs/specification/frozen/v1.0.0/SPECIFICATION.md`  
**License Register:** `docs/specification/dataset/DATASET_LICENSE_REGISTER.md`  
**Candidate Matrix:** `docs/specification/audits/GATE_2_DATASET_CANDIDATE_MATRIX.md`

---

## Design Principles

1. **DATA QUALITY > DATASET SIZE** — The historical 10.23M-row dataset failed Gate 1. A smaller, well-curated corpus is preferable.
2. **Only Grade A datasets in the primary pipeline** — No legal/license ambiguity.
3. **Provenance is mandatory** — Every row must carry `SOURCE`/`A`/`B`/`C` provenance per SPEC-04.
4. **Labels are never invented** — If a source lacks severity labels, that field is `NULL`, not heuristic-filled (unless explicitly tagged as `regex_heuristic` in `severity_source`).
5. **Split integrity before augmentation** — Splits are assigned based on `seed_id` before any augmentation step.
6. **LLM-generated data must never dominate** — Category C text is capped and always disclosed.
7. **Multilingual coverage comes from legitimate sources + deterministic augmentation** — Not from mislabeled foreign-language corpora.

---

## A. CORE CLINICAL SOURCES

These sources provide genuine clinical text for primary supervised training.

### A1. MTSamples

| Field | Value |
|---|---|
| **Role** | PRIMARY_TRAINING |
| **Provenance** | SOURCE |
| **Est. Rows** | ~2,400 |
| **Language** | en |
| **Department Labels** | Yes — via `src/specialty_mapping.py` (high-confidence, human-curated mapping) |
| **Severity Labels** | None |
| **Rationale** | Small but anchor-quality clinical transcriptions covering diverse medical specialties. Only source with human-verified department mapping. CC0 license. |
| **Adapter** | `meditriage/builder/adapters/mtsamples.py` (EXISTS, needs schema alignment) |

### A2. NEISS (Downsampled)

| Field | Value |
|---|---|
| **Role** | PRIMARY_TRAINING |
| **Provenance** | SOURCE |
| **Est. Rows** | Cap at **500,000** (from 7.1M historical, ~7% sample) |
| **Language** | en |
| **Department Labels** | Yes — heuristic (diagnosis code + body part + narrative regex; adapter-inferred) |
| **Severity Labels** | None |
| **Rationale** | Public domain CPSC injury narratives. Provides real ED text at scale. **Must be downsampled** to prevent the 70% dominance observed in the historical dataset. The PEDS override (age<18 → PEDS for all departments) must be reconsidered. |
| **Downsampling Strategy** | Stratified by department to preserve class distribution. Random seed = 42 for reproducibility. |
| **Adapter** | `meditriage/builder/adapters/neiss.py` (EXISTS, works correctly) |

### A3. NHAMCS ED

| Field | Value |
|---|---|
| **Role** | PRIMARY_TRAINING |
| **Provenance** | SOURCE |
| **Est. Rows** | ~41,500 |
| **Language** | en |
| **Department Labels** | ED only (all records are ED visits by definition) |
| **Severity Labels** | **Yes — ESI 1–5** (genuine triage levels from IMMEDR field) |
| **Rationale** | CDC public-use data with real ESI triage labels. One of only two clean sources with genuine severity. Critical for severity-head training. |
| **Adapter** | `meditriage/builder/adapters/nhamcs_ed.py` (EXISTS, works correctly) |

### A4. PMC-Patients

| Field | Value |
|---|---|
| **Role** | SECONDARY_TRAINING |
| **Provenance** | SOURCE |
| **Est. Rows** | ~167,000 |
| **Language** | en |
| **Department Labels** | Inferred (adapter uses keyword classification) |
| **Severity Labels** | None |
| **Rationale** | Real clinical case reports from PubMed Central. High-quality, diverse clinical narratives. **CC BY-NC-SA 4.0** — research use only. Must carry license metadata. |
| **Adapter** | `meditriage/builder/adapters/pmc_patients.py` (EXISTS, works correctly) |
| **License Constraint** | Non-commercial, ShareAlike. Compliant with research use. |

---

## B. LANGUAGE / MULTILINGUAL SOURCES

### B1. L3Cube HingLID (Hinglish Linguistic Patterns)

| Field | Value |
|---|---|
| **Role** | AUXILIARY_LANGUAGE |
| **Provenance** | SOURCE |
| **Est. Rows** | ~37,700 |
| **Language** | hi-en (Hinglish, code-mixed) |
| **Department Labels** | Adapter-inferred (Hinglish keyword mapping) |
| **Severity Labels** | None |
| **Rationale** | MIT-licensed token-level language-ID dataset. NOT medical content — provides Hinglish code-mixing linguistic patterns. Used as reference for building Hinglish augmentation, not as direct clinical training data. |
| **Usage** | Feed into the multilingual augmentation pipeline as linguistic pattern reference. NOT counted as a primary clinical source. |
| **Adapter** | `meditriage/builder/adapters/l3cube_code_mixed.py` (EXISTS, works correctly) |

### B2. Deterministic Hinglish/Romanized Hindi Augmentation (Pipeline-Generated)

| Field | Value |
|---|---|
| **Role** | AUXILIARY_LANGUAGE |
| **Provenance** | A (deterministic linguistic augmentation) |
| **Est. Rows** | TBD (function of augmentation budget) |
| **Language** | hi-Latn, hi-en, en-hi |
| **Rationale** | The multilingual variation engine (`meditriage/multilingual/variation/`) provides 10 clinical variation styles with deterministic phonetic/lexical transformations. This is the primary mechanism for generating Hinglish/romanized training data from English clinical text. |
| **Key Constraint** | All augmented rows must carry `provenance=A`, `augmentation_parent_id` pointing to the source record, and never cross split boundaries. |
| **Infrastructure** | `meditriage/multilingual/` (EXISTS — translator, validator, variation engine, providers) |

---

## C. ROBUSTNESS SOURCES / STRATA

### C1. Clinical Linguistic Variation Engine (Pipeline-Generated)

| Field | Value |
|---|---|
| **Role** | ROBUSTNESS_ONLY |
| **Provenance** | A |
| **Variation Styles** | Lexical, syntactic, conversational, ED triage, physician note, nurse intake, patient spoken, abbreviated clinical, formal documentation, colloquial Indian expression |
| **Infrastructure** | `meditriage/multilingual/variation/generators.py` (EXISTS — 10 generators implemented) |

### C2. Symptom2Disease

| Field | Value |
|---|---|
| **Role** | SECONDARY_TRAINING + ROBUSTNESS_ONLY |
| **Provenance** | SOURCE |
| **Est. Rows** | ~1,200 |
| **Rationale** | Small symptom-to-disease mapping dataset. Useful for ensuring the model recognizes common symptom patterns. CC0. |

### C3. Kaggle Medical Triage

| Field | Value |
|---|---|
| **Role** | SECONDARY_TRAINING |
| **Provenance** | SOURCE |
| **Est. Rows** | ~1,100 |
| **Severity Labels** | Yes — string format (Routine/Urgent/Emergency/Observation) requiring mapping |
| **Rationale** | Small but has triage labels. CC0. String severity must be mapped to ESI scale. |

---

## D. SAFETY / RED-FLAG SOURCES

### D1. Existing OOD Queries

| Field | Value |
|---|---|
| **Role** | SAFETY_EVALUATION_ONLY |
| **Source** | `data/ood_queries.csv` |
| **Provenance** | SOURCE |
| **Est. Rows** | ~1,000+ (to be audited) |
| **Rationale** | Existing OOD query set for safety evaluation. Must NOT enter training. |

### D2. Red-Flag Evaluation Dataset (To Be Built)

| Field | Value |
|---|---|
| **Role** | SAFETY_EVALUATION_ONLY |
| **Provenance** | B (rule-based/templated construction) |
| **Status** | DECIDED to build (per SPEC-07), size/strata TBD |
| **Rationale** | Required by frozen specification. Strata defined in SPEC-07 Section 10. |

---

## E. OOD SOURCES

### E1. MedQA-USMLE

| Field | Value |
|---|---|
| **Role** | REFERENCE_ONLY |
| **Provenance** | SOURCE |
| **Rationale** | Medical exam Q&A format — completely different from triage input format. Useful as OOD reference to test whether model gracefully handles non-triage medical text. |

### E2. Medical Meadow MedQA

| Field | Value |
|---|---|
| **Role** | REFERENCE_ONLY |
| **Provenance** | SOURCE |
| **Rationale** | Same as MedQA-USMLE — Q&A format, no triage labels, useful as OOD reference only. |

---

## F. OPTIONAL / AUXILIARY SOURCES

### F1. Triagegeist (Not Yet Acquired)

| Field | Value |
|---|---|
| **Role** | AUXILIARY (if acquired) |
| **Provenance** | C (synthetic) |
| **License** | CC0 (Kaggle) |
| **Rationale** | Synthetic ED dataset with ESI labels, demographics, vitals, chief complaints. Would provide additional severity-labeled training data. Must be explicitly marked as Category C. Acquisition deferred to Phase B. |

---

## G. REJECTED SOURCES

| Source | Reason |
|---|---|
| **ChatDoctor HealthCareMagic** | License Grade E — scraped commercial data without verified consent. **PERMANENTLY REJECTED.** |
| **ChatDoctor iCliniq** | License Grade E — scraped commercial platform data. **PERMANENTLY REJECTED.** |
| **MedDialog EN** | License Grade D + 100% CJK content mislabeled as English. **QUARANTINED pending investigation.** |
| **FedMML ED Triage** | License Grade D + 100% LLM-generated. **QUARANTINED pending license clarification.** |

---

## Historical Dataset Disposition

**`meditriage/data/processed/dataset.parquet`** (10,230,264 rows)

**Decision: NOT ELIGIBLE for canonical use. ARCHIVED for reference.**

Rationale:
1. Failed 10/18 DATASET-GATE-01 requirements (Gate 1 report)
2. 99.63% English, 0% Hindi/Devanagari, 0% Romanized Hindi
3. 25.58% CJK-contaminated rows (mislabeled Chinese)
4. 1.17% scraped commercial data (ChatDoctor)
5. 0.82% license-unknown synthetic data (FedMML)
6. Stage 5 augmentation was a no-op (file copy)
7. Missing provenance metadata fields
8. No `text_provenance` or `label_provenance` columns
9. 98.89% missing severity labels
10. Severe class imbalance (top 4 departments = 87%)

The file remains on disk for reference/comparison but is **not the input to the canonical pipeline**.

---

## Estimated Canonical Dataset Composition

| Component | Role | Est. Rows (Pre-Augmentation) | Severity Labels | Department Labels |
|---|---|---|---|---|
| MTSamples | PRIMARY_TRAINING | ~2,400 | 0% | 100% |
| NEISS (downsampled) | PRIMARY_TRAINING | ~500,000 | 0% | 100% |
| NHAMCS ED | PRIMARY_TRAINING | ~41,500 | **100%** | 100% (ED) |
| PMC-Patients | SECONDARY_TRAINING | ~167,000 | 0% | ~95% (inferred) |
| Symptom2Disease | SECONDARY_TRAINING | ~1,200 | 0% | 100% |
| Kaggle Medical Triage | SECONDARY_TRAINING | ~1,100 | **100%** | Partial |
| L3Cube HingLID | AUXILIARY_LANGUAGE | ~37,700 | 0% | Inferred |
| Augmentation (Stage 5) | ROBUSTNESS | TBD | Preserved | Preserved |
| **Pre-augmentation total** | — | **~751,000** | ~5.7% | ~99.8% |

**Post-augmentation estimate** (with multilingual expansion, variation engine): TBD — dependent on augmentation budget configuration. Expected 2x–5x expansion for targeted rows.

> **NOTE:** The canonical dataset will be significantly smaller than the historical 10.23M rows. This is intentional — DATA QUALITY > DATASET SIZE. The composition is cleaner, legally clear, and properly labeled.
