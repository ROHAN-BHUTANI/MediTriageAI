# MediTriageAI — Repository Modularization Plan

**Specification Baseline:** `v1.0.0-FROZEN`  
**Document Status:** MODULARIZATION ROADMAP  
**Date:** `2026-08-16`

---

## 1. Classification Framework

This plan classifies all modules across the codebase into four strict tiers to guide future refactoring without risking specification violations, data corruption, or test breakages.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           MODULARIZATION TIERS                          │
├───────────────────┬─────────────────────────────────────────────────────┤
│ TIER 1: SAFE NOW  │ Dead imports, one-off temporary files, clean unused │
│                   │ scratch scripts with 0 inbound references.          │
├───────────────────┼─────────────────────────────────────────────────────┤
│ TIER 2: SAFE      │ Deprecated legacy transforms in src/transforms/     │
│ AFTER TESTS       │ whose functionality is 100% in meditriage/.         │
├───────────────────┼─────────────────────────────────────────────────────┤
│ TIER 3: REQUIRES  │ Unifying src/trainer.py and meditriage/training/    │
│ MIGRATION         │ into a single clean training package.               │
├───────────────────┼─────────────────────────────────────────────────────┤
│ TIER 4: DO NOT    │ Frozen specifications, baseline datasets,           │
│ TOUCH (IMMUTABLE) │ verified canonical datasets, model architectures.   │
└───────────────────┴─────────────────────────────────────────────────────┘
```

---

## 2. Action Items by Tier

### Tier 1: Safe Now
- **Actions:**
  - Removed process-dependent `hash(parent_id)` and replaced with deterministic `stable_seed(parent_id)` in `scripts/build_canonical.py`.
  - Added robust `_is_null()` handling in `meditriage/builder/canonical_schema.py`.
  - Added automated `scripts/flight_check.py`.
  - Retain temporary scripts in `scratch/` for historical diagnostic continuity without active imports.

### Tier 2: Safe After Test Suite Expansion
- **Modules:**
  - `src/transforms/*.py` (12 files)
  - `src/clinical_safety_validator.py`
  - `src/duplicate_validator.py`
  - `src/diversity_scorer.py`
- **Plan:** These modules have 0 inbound imports from production scripts. When the full test suite is migrated to test `meditriage/` directly, move these to `archive/legacy_src/`.

### Tier 3: Requires Controlled Migration (Post-Training Campaign)
- **Target:** Convergence of `src/` into `meditriage/`:
  - `src/trainer.py` → `meditriage/training/trainer.py`
  - `src/data_pipeline.py` → `meditriage/training/dataset.py`
  - `src/metrics.py` → `meditriage/training/metrics.py`
  - `src/model.py` → `meditriage/models/`
- **Constraint:** Migration must only occur after benchmark training runs on the frozen v1.0.0 baseline are completed and validated.

### Tier 4: Do Not Touch (Immutable Governance)
- `docs/specification/frozen/v1.0.0/**` (Permanently frozen)
- `docs/specification/audits/GATE_1_HISTORICAL_LANGUAGE_AUDIT.md`
- `docs1/specification/_freeze_source/**`
- `meditriage/data/processed/dataset.parquet` (Historical baseline dataset)
- `meditriage/data/canonical/v1.0.0/dataset.parquet` (Verified canonical dataset)
- `meditriage/data/canonical/v1.0.0/build_manifest.json` (Verified build manifest)
