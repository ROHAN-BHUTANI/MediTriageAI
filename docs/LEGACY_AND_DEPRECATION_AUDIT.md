# MediTriageAI — Legacy Code & Deprecation Audit

**Specification Baseline:** `v1.0.0-FROZEN`  
**Audit Date:** `2026-08-16`  
**Governing ADRs:** ADR-003, ADR-004, ADR-012

---

## 1. Dead Code Audit (`src/` Subsystem)

Per SPEC-03 / SPEC-10, modules in `src/` with zero inbound imports are identified for safe archival.

| File Path | Inbound References | Last Known Purpose | Safe to Archive? | Replacement Path | Dependency Risk |
|---|---|---|---|---|---|
| `src/clinical_safety_validator.py` | 0 | Historical regex-based clinical safety check | **YES** | `meditriage/builder/canonical_schema.py` | ZERO |
| `src/diversity_scorer.py` | 0 | Historical embedding diversity scoring | **YES** | Subsumed by stratified split | ZERO |
| `src/duplicate_validator.py` | 0 | Historical duplicate checker | **YES** | `scripts/build_canonical.py::deduplicate` | ZERO |
| `src/experiment_manager.py` | 0 | Historical MLflow / experiment tracking wrapper | **YES** | `src/evaluation.py` | ZERO |
| `src/leakage_safe_split.py` | 0 | Historical hash splitter | **YES** | `scripts/build_canonical.py::assign_stratified_splits` | ZERO |
| `src/registry.py` | 0 | Historical dataset adapter registry | **YES** | `meditriage/builder/canonical_schema.py` | ZERO |
| `src/severity_heuristic.py` | 0 | Deprecated regex-based severity heuristic | **YES** | Native ESI labels in canonical schema | ZERO |
| `src/transformation_base.py` | 0 | Historical base class for transforms | **YES** | `meditriage/multilingual/variation/` | ZERO |
| `src/transforms/abbreviation_compression.py` | 0 | Transform plugin | **YES** | `meditriage/multilingual/variation/` | ZERO |
| `src/transforms/abbreviation_expansion.py` | 0 | Transform plugin | **YES** | `meditriage/multilingual/variation/` | ZERO |
| `src/transforms/asr_noise.py` | 0 | Transform plugin | **YES** | `scripts/build_canonical.py::generate_asr_noise` | ZERO |
| `src/transforms/case_variation.py` | 0 | Transform plugin | **YES** | `meditriage/multilingual/variation/` | ZERO |
| `src/transforms/clinical_shorthand.py` | 0 | Transform plugin | **YES** | `meditriage/multilingual/variation/` | ZERO |
| `src/transforms/duration_variation.py` | 0 | Transform plugin | **YES** | `meditriage/multilingual/variation/` | ZERO |
| `src/transforms/number_formatting.py` | 0 | Transform plugin | **YES** | `meditriage/multilingual/variation/` | ZERO |
| `src/transforms/punctuation_variation.py` | 0 | Transform plugin | **YES** | `meditriage/multilingual/variation/` | ZERO |
| `src/transforms/synonym_replacement.py` | 0 | Transform plugin | **YES** | `meditriage/multilingual/variation/` | ZERO |
| `src/transforms/typo_generator.py` | 0 | Transform plugin | **YES** | `meditriage/multilingual/variation/` | ZERO |
| `src/transforms/unit_conversion.py` | 0 | Transform plugin | **YES** | `meditriage/multilingual/variation/` | ZERO |
| `src/transforms/vague_complaint.py` | 0 | Transform plugin | **YES** | `meditriage/multilingual/variation/` | ZERO |

---

## 2. Superseded / Experimental Subsystems

| Subsystem | Location | Current State | Recommendation |
|---|---|---|---|
| `reconstruction/` | `reconstruction/` (26 files) | 10-stage historical dataset reconstruction pipeline from prior research sprint | **PRESERVE / DO NOT TOUCH** (Historic research artifact; do not delete or execute) |
| `ref/` | `ref/` (27 files) | Abstract benchmark framework | **PRESERVE / DO NOT TOUCH** |
| `results/multilingual_forensic/` | `results/multilingual_forensic/` (29 files) | Gate 1 historical forensic investigation artifacts | **PRESERVE / DO NOT TOUCH** |

---

## 3. Temporary Scratch Artifact Classification (Phase 6)

| Path | Description | Classification | Policy |
|---|---|---|---|
| `scratch/*.py` | One-off audit & diagnostic inspection scripts from previous sessions | **KEEP IN SCRATCH** | Retain for diagnostic history; do not reference in production |
| `.planning/` | Planning documents and task logs | **KEEP** | Standard project coordination artifacts |
| `meditriage/data/processed/dataset.parquet` | Historical 10.23M-row baseline dataset | **DO NOT TOUCH (IMMUTABLE)** | Governed baseline dataset |
| `meditriage/data/canonical/v1.0.0/` | Canonical production dataset & manifests | **PRIMARY DATASET** | Governed canonical production dataset |
