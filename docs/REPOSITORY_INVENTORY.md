# MediTriageAI — Complete Repository Inventory

**Specification Baseline:** `v1.0.0-FROZEN`  
**Audit Date:** `2026-08-16`  
**Repository State:** Post-Canonical Build & Flight Check

---

## 1. Executive Summary

A comprehensive, ground-truth inventory of all subsystems across the `MediTriageAI_Data_Engine` repository. A total of 1,588 files across 21 subsystems were mapped, categorized, and audited for operational criticality, dependency relationships, and migration status.

| Subsystem / Directory | Files | Total LOC | Primary Role | Criticality Tier |
|---|---|---|---|---|
| `scripts/` | 28 | 10,965 | Primary execution entrypoints & utilities | **TIER 1 — PRODUCTION / PIPELINE** |
| `meditriage/builder/` | 31 | 16,100 | Canonical data engine & ingestion adapters | **TIER 1 — PRODUCTION / PIPELINE** |
| `meditriage/multilingual/` | 31 | 4,405 | Linguistic variation & code-mixing engine | **TIER 1 — PRODUCTION / PIPELINE** |
| `meditriage/training/` | 15 | 1,993 | Target modular training harness (`meditriage/`) | **TIER 1 — TRAINING TARGET** |
| `models/` | 28 | 5,570 | Transformer backbones & E-PATH modules | **TIER 1 — MODEL ZOO** |
| `src/` | 41 | 6,092 | Proven legacy training harness & transforms | **TIER 1 — ACTIVE HARNESS** |
| `tests/` | 49 | 10,312 | Pytest test suites (unit, integration, pipeline) | **TIER 1 — VERIFICATION** |
| `docs/` | 44 | 4,521 | Architecture specs, audits, runbooks, governance | **TIER 2 — GOVERNANCE / DOCS** |
| `datasets/` | 77 | 201,795 | Raw and cached source datasets | **TIER 2 — DATA ASSETS** |
| `analysis/` | 13 | 2,945 | Post-hoc statistical evaluation & calibration | **TIER 2 — EVALUATION** |
| `reconstruction/` | 26 | 3,744 | Historical dataset reconstruction stages | **TIER 3 — EXPERIMENTAL / LEGACY** |
| `ref/` | 27 | 3,575 | Benchmark and experiment runner references | **TIER 3 — REFERENCE** |
| `results/` | 127 | 85,334 | Checkpoint metrics, logs, forensic audits | **TIER 3 — RUN ARTIFACTS** |
| `paper_artifacts/` | 26 | 126 | Figures, tables, and publication diagrams | **TIER 3 — PUBLICATION** |
| `scratch/` | 25 | 3,274 | Temporary inspection and audit scripts | **TIER 4 — SCRATCH / AUDIT** |

---

## 2. Detailed Subsystem Mapping

### 2.1. Executable Scripts (`scripts/`)
| Script | Subsystem | Purpose | Dependencies | Criticality | Status |
|---|---|---|---|---|---|
| `scripts/build_canonical.py` | Pipeline | 20-stage full canonical dataset builder | `meditriage.builder`, `pyarrow`, `pandas` | Production-Critical | **PRIMARY** |
| `scripts/build_pilot.py` | Pipeline | Representative pilot dataset builder | `meditriage.builder`, `pyarrow`, `pandas` | Verification-Critical | **PRIMARY PILOT** |
| `scripts/flight_check.py` | Quality Gate | Automated pre-training flight check | `pyarrow`, `pytest`, `torch` | Pre-flight Critical | **PRIMARY GATE** |
| `scripts/train.py` | Training | Local single-GPU training entrypoint | `src.trainer`, `src.model`, `src.dataset` | Training-Critical | **PRIMARY TRAIN** |
| `scripts/train_ddp.py` | Training | Multi-GPU Distributed Data Parallel training | `src.trainer`, `src.model`, `torch.distributed` | Training-Critical | **PRIMARY DDP** |
| `scripts/colab_train.py` | Training | Single-GPU cloud training wrapper | `src.trainer`, `src.model` | Secondary | ACTIVE |
| `scripts/evaluate.py` | Evaluation | Full model evaluation & metric calculation | `src.evaluation`, `src.metrics`, `src.model` | Eval-Critical | **PRIMARY EVAL** |
| `scripts/dataset_audit.py` | Audit | Standalone dataset statistics & class entropy | `src.metrics`, `pandas` | Secondary | ACTIVE |
| `scripts/error_analysis.py` | Evaluation | Error breakdown by department and severity | `src.metrics`, `pandas` | Secondary | ACTIVE |
| `scripts/multilingual_expansion.py` | Augmentation | Offline multilingual expansion script | `meditriage.multilingual` | Auxiliary | ACTIVE |
| `scripts/phenotype_augmentation.py` | Augmentation | Clinical phenotype expansion script | `meditriage.multilingual.phenotype` | Experimental | ACTIVE |
| `scripts/hard_negative_generation.py` | Augmentation | Clinical hard-negative generation script | `meditriage.multilingual.hard_negative` | Experimental | ACTIVE |

### 2.2. Core Data Engine (`meditriage/builder/`)
| Module | Responsibility | Consumers | Status |
|---|---|---|---|
| `meditriage/builder/canonical_schema.py` | 26-field canonical PyArrow schema & enum validators | `build_canonical.py`, `build_pilot.py`, tests | **AUTHORITATIVE** |
| `meditriage/builder/adapters/mtsamples.py` | MTSamples raw CSV ingestion adapter | `meditriage.builder.orchestrator` | ACTIVE |
| `meditriage/builder/adapters/neiss.py` | NEISS raw Parquet ingestion adapter | `meditriage.builder.orchestrator` | ACTIVE |
| `meditriage/builder/adapters/nhamcs_ed.py` | NHAMCS ED fixed-width raw ingestion adapter | `meditriage.builder.orchestrator` | ACTIVE |
| `meditriage/builder/adapters/symptom2disease.py` | Symptom2Disease raw CSV adapter | `meditriage.builder.orchestrator` | ACTIVE |
| `meditriage/builder/adapters/kaggle_medical_triage.py`| Kaggle Medical Triage Parquet adapter | `meditriage.builder.orchestrator` | ACTIVE |
| `meditriage/builder/adapters/chatdoctor.py` | ChatDoctor ingestion adapter (Grade E) | Excluded by License Gate | **QUARANTINED** |
| `meditriage/builder/adapters/fedmml.py` | FedMML synthetic ED adapter (Grade D) | Excluded by License Gate | **QUARANTINED** |
| `meditriage/builder/adapters/meddialog.py` | MedDialog EN adapter (Grade D CJK) | Excluded by License Gate | **QUARANTINED** |

### 2.3. Multilingual & Robustness Subsystem (`meditriage/multilingual/`)
| Module | Responsibility | Consumers | Status |
|---|---|---|---|
| `meditriage/multilingual/variation/generators.py` | Clinical linguistic variation (lexical, shorthand, informal, colloquial Indian) | `build_canonical.py`, tests | **ACTIVE** |
| `meditriage/multilingual/variation/engine.py` | Variation pipeline orchestration | `scripts/multilingual_expansion.py` | ACTIVE |
| `meditriage/multilingual/providers/offline.py` | Rule-based offline multilingual transliteration & Hindi terms | `build_canonical.py`, tests | **ACTIVE** |
| `meditriage/multilingual/hard_negative/hard_negative_library.py` | Clinical hard-negative differential definitions | `build_canonical.py`, tests | **ACTIVE** |
| `meditriage/multilingual/phenotype/phenotype_library.py` | Clinical phenotype pattern definitions | `scripts/phenotype_augmentation.py` | RESEARCH |

### 2.4. Model Subsystems (`models/` and `src/model.py`)
| Module | Description | Backbones Supported | Status |
|---|---|---|---|
| `src/model.py` | Canonical `MediTriageTransformer` dual-head architecture (13 depts, 5 ESI) | XLM-RoBERTa, MuRIL, mBERT, DistilBERT | **ACTIVE PRODUCTION** |
| `models/base_model.py` | Subclass interface for `meditriage/` model zoo | Generic transformer | MIGRATION |
| `models/xlm_roberta.py` | `XLMRobertaLargeModel` wrapper (`xlm-roberta-base` checkpoint) | `xlm-roberta-base` | ACTIVE |
| `models/indic_bert.py` | `IndicBertModel` wrapper (`google/muril-base-cased` checkpoint) | `google/muril-base-cased` | ACTIVE |
| `models/mbert.py` | `MBertModel` wrapper (`bert-base-multilingual-cased`) | `bert-base-multilingual-cased` | ACTIVE |
| `models/distilbert_multi.py` | `DistilBertMultilingualModel` wrapper | `distilbert-base-multilingual-cased` | ACTIVE |
| `models/emergent_path_triage/` | E-PATH clinical co-reasoning architecture (DCCF, AMCO, DCES, DCRR, CTB) | Multi-aspect co-reasoning | **RESEARCH-EXPERIMENTAL** |

### 2.5. Training & Evaluation Harnesses (`src/` vs `meditriage/training/`)
| Module | Subsystem | Responsibility | Status |
|---|---|---|---|
| `src/trainer.py` | `src/` | Proven multi-task trainer with masked Focal Loss | **ACTIVE BINDING** |
| `src/data_pipeline.py` | `src/` | PyTorch Dataset & DataLoader constructor | **ACTIVE BINDING** |
| `src/specialty_mapping.py` | `src/` | 13-department taxonomy and raw string mapping | **AUTHORITATIVE** |
| `src/vocab_injection.py` | `src/` | Token embedding expansion for Hinglish variants | **ACTIVE** |
| `src/metrics.py` | `src/` | Macro-F1, AUROC, ECE, MAE, ordinal confusion | **AUTHORITATIVE** |
| `src/evaluation.py` | `src/` | `EvaluationExporter` structured JSON output | **ACTIVE** |
| `meditriage/training/trainer.py` | `meditriage/` | Modular training harness target | TARGET (Migration Priority 1) |
| `meditriage/training/losses.py` | `meditriage/` | Masked Focal Loss & joint loss functions | TARGET (Migration Priority 1) |

---

## 3. Legacy and Dead Module Register

Per SPEC-03 / SPEC-10 audit criteria, the following modules in `src/` have zero inbound references and are classified safe for controlled archiving:

1. `src/clinical_safety_validator.py` (Dead, 0 references)
2. `src/diversity_scorer.py` (Dead, 0 references)
3. `src/duplicate_validator.py` (Dead, 0 references)
4. `src/experiment_manager.py` (Dead, 0 references)
5. `src/leakage_safe_split.py` (Dead, replaced by `assign_stratified_splits`)
6. `src/registry.py` (Dead, 0 references)
7. `src/severity_heuristic.py` (Dead, regex heuristic deprecated)
8. `src/transformation_base.py` (Dead, 0 references)
9. `src/transforms/*.py` (12 dead files; replaced by `meditriage/multilingual/variation/`)

---

## 4. Preservation Invariants

The following paths are permanently immutable under governance:
- `docs/specification/frozen/v1.0.0/**` (Frozen baseline contract)
- `docs/specification/audits/GATE_1_HISTORICAL_LANGUAGE_AUDIT.md` (Gate 1 empirical audit)
- `docs1/specification/_freeze_source/**` (Original Claude freeze source documents)
- `meditriage/data/processed/dataset.parquet` (Historical baseline dataset, 1.3 GB)
