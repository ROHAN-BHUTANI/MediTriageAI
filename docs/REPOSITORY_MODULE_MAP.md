# MediTriageAI — Subsystem Ownership & Module Map

**Specification Baseline:** `v1.0.0-FROZEN`  
**Document Status:** AUTHORITATIVE ARCHITECTURE MAP  
**Date:** `2026-08-16`

---

## 1. Subsystem Ownership Boundaries

This document defines the 12 functional subsystem boundaries for MediTriageAI, establishing strict module responsibilities, entry points, dependencies, and forbidden couplings.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            1. DATA INGESTION                                │
│           datasets/raw/ → meditriage/builder/adapters/                      │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        2. CANONICAL DATASET BUILD                           │
│     scripts/build_canonical.py → meditriage/builder/canonical_schema.py     │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                 3. MULTILINGUAL & ROBUSTNESS AUGMENTATION                   │
│          meditriage/multilingual/ (variation, offline, hard_negative)       │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      4. DATA QUALITY & GATE VALIDATION                      │
│     scripts/flight_check.py → tests/test_canonical_pipeline.py              │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       5. MODEL ARCHITECTURE & ZOO                           │
│                 src/model.py ↔ models/ (backbone wrappers)                  │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                             6. MODEL TRAINING                               │
│            scripts/train.py / scripts/train_ddp.py → src/trainer.py         │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            7. EVALUATION & METRICS                          │
│               scripts/evaluate.py → src/metrics.py, src/evaluation.py       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Module Ownership Matrix

### 1. Data Ingestion Subsystem
- **Path:** `meditriage/builder/adapters/`
- **Responsibility:** Ingest upstream raw formats (CSV, fixed-width, Parquet) into standardized field dictionaries.
- **Public Entry Points:** `ingest_mtsamples()`, `ingest_neiss()`, `ingest_nhamcs_ed()`, `ingest_kaggle_triage()`, `ingest_symptom2disease()`.
- **Dependencies:** `pandas`, `pyarrow`, `json`.
- **Consumers:** `scripts/build_canonical.py`, `scripts/build_pilot.py`.
- **Forbidden Responsibilities:** Must not invent missing labels, must not map Kaggle urgency to ESI, must not assign splits.

### 2. Canonical Dataset Build Subsystem
- **Path:** `scripts/build_canonical.py`, `meditriage/builder/canonical_schema.py`
- **Responsibility:** Ingest, normalize, clean, deduplicate, stratify into 80/10/10 splits, augment, and export canonical Parquet with SHA-256 manifests.
- **Public Entry Points:** `scripts/build_canonical.py::main()`.
- **Dependencies:** `pyarrow`, `meditriage.builder.canonical_schema`.
- **Consumers:** Training pipeline, evaluation pipeline, benchmark harness.
- **Forbidden Responsibilities:** Must not alter frozen specifications; must not overwrite `meditriage/data/processed/dataset.parquet`.

### 3. Data Quality & Validation Subsystem
- **Path:** `scripts/flight_check.py`, `tests/test_canonical_pipeline.py`
- **Responsibility:** Automated verification of 26-field schema, non-null constraints, enums, zero leakage, checksum matching, and DATASET-GATE-01 audit.
- **Public Entry Points:** `python scripts/flight_check.py`, `pytest tests/test_canonical_pipeline.py`.
- **Dependencies:** `pytest`, `pyarrow`, `torch`.
- **Consumers:** CI/CD, pre-training gates, DGX pre-launch checks.
- **Forbidden Responsibilities:** Must not silently skip failing assertions or downgrade errors to warnings.

### 4. Multilingual & Robustness Subsystem
- **Path:** `meditriage/multilingual/`
- **Responsibility:** Provide deterministic in-text linguistic transforms (lexical, shorthand, informal, colloquial Indian, ASR noise, Hinglish, Devanagari Hindi, Roman Hindi, hard negatives, late red flags).
- **Public Entry Points:** `generate_multilingual_variants()`, `generate_asr_noise()`, `generate_hard_negatives()`, `generate_late_red_flag()`.
- **Dependencies:** Standard library (`random`, `re`, `hashlib`).
- **Consumers:** `scripts/build_canonical.py`.
- **Forbidden Responsibilities:** Must not alter clinical negation, duration, severity, or department semantics.

### 5. Model Architecture Subsystem
- **Path:** `src/model.py`, `models/`
- **Responsibility:** Multi-task dual-head transformer architecture (`Linear(hidden_size, 13)` specialist head + `Linear(hidden_size, 5)` severity head from shared `[CLS]` token).
- **Public Entry Points:** `MediTriageTransformer`, `XLMRobertaLargeModel`, `IndicBertModel`, `MBertModel`, `DistilBertMultilingualModel`.
- **Dependencies:** `torch`, `transformers`.
- **Consumers:** Training and evaluation harnesses.
- **Forbidden Responsibilities:** Must not hardcode checkpoint name aliases; must not pool experimental E-PATH weights with baseline backbones.

### 6. Training Subsystem
- **Path:** `scripts/train.py`, `scripts/train_ddp.py`, `src/trainer.py`, `src/data_pipeline.py`
- **Responsibility:** Execute local and multi-GPU DDP training with masked Focal Loss (`ignore_index=-1`), early stopping, learning rate scheduling, and checkpointing.
- **Public Entry Points:** `python scripts/train.py`, `torchrun --nproc_per_node=N scripts/train_ddp.py`.
- **Dependencies:** `torch`, `src.model`, `src.data_pipeline`, `src.config_manager`.
- **Consumers:** DGX training campaigns, researcher experiments.
- **Forbidden Responsibilities:** Must not train without verified dataset checksum matching the manifest.

### 7. Evaluation Subsystem
- **Path:** `scripts/evaluate.py`, `src/metrics.py`, `src/evaluation.py`
- **Responsibility:** Calculate primary Macro-F1, top-k accuracy, AUROC, ECE, severity MAE, ordinal confusion, and export version-tagged evaluation JSON envelopes.
- **Public Entry Points:** `python scripts/evaluate.py`.
- **Dependencies:** `src.metrics`, `src.evaluation`, `scikit-learn`, `numpy`.
- **Consumers:** Benchmark reports, paper generation scripts, dashboard.
- **Forbidden Responsibilities:** Must not handcraft or retrospectively pick primary metrics.

### 8. Experiments & Research Subsystem
- **Path:** `models/emergent_path_triage/`, `analysis/`
- **Responsibility:** Research implementations including E-PATH co-reasoning (AMCO, DCCF, DCES, DCRR, CTB) and error analysis.
- **Public Entry Points:** `pytest tests/test_emergent_path_triage.py`, `python scripts/error_analysis.py`.
- **Dependencies:** `torch`, `models.emergent_path_triage`.
- **Consumers:** Ablation studies and research paper sections.
- **Forbidden Responsibilities:** Must not replace production baseline models without explicit change request.

### 9. Testing Subsystem
- **Path:** `tests/`
- **Responsibility:** Comprehensive test coverage across data pipeline, schema, model zoo, metrics, loss masking, and integration.
- **Public Entry Points:** `pytest`.
- **Dependencies:** `pytest`, `anyio`, `torch`.
- **Consumers:** CI/CD and developer verification.

### 10. Deployment & UI Subsystem
- **Path:** `src/dashboard.py` (legacy Streamlit UI), `scripts/paper/`
- **Responsibility:** Visualization of triage predictions, research demo interface, and publication table/figure generation.
- **Public Entry Points:** `streamlit run src/dashboard.py`, `python scripts/paper/tables.py`.
- **Dependencies:** `streamlit`, `matplotlib`.
- **Consumers:** Reviewers, demo users, paper compilation.
- **Forbidden Responsibilities:** Must never display clinical predictions without the mandatory byte-identical disclaimer.

### 11. Documentation Subsystem
- **Path:** `docs/`, `docs1/`
- **Responsibility:** Specifications, frozen baseline contract, audit records, architecture maps, and flight checklists.
- **Consumers:** Engineers, AI assistants, auditors.

### 12. Archived & Legacy Subsystem
- **Path:** `reconstruction/`, `ref/`, `src/transforms/`
- **Responsibility:** Historical references, superseded reconstruction prototypes, and deprecated regex heuristics.
- **Status:** READ-ONLY / SAFE FOR ARCHIVAL.
