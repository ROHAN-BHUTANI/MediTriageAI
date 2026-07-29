# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased] - Final Pre-Training Setup

### Added
- Comprehensive Dataset Registry (`src/dataset_registry.py`).
- Split-aware Dataset Adapters (`src/dataset_adapters.py`) supporting NHAMCS, ChatDoctor (HealthCareMagic & iCliniq), L3Cube, and PMC-Patients.
- Unified Dataset Builder (`src/dataset_builder.py`) for in-memory dataset aggregation.
- Validation scripts for robust external data parsing (`scripts/validate_datasets.py`).
- Multiple detailed architectural reports (`DATASET_INTEGRATION_REPORT.md`, `DATASET_ENRICHMENT_REPORT.md`, `TRAINING_STRATEGY_REPORT.md`, `RESEARCH_READINESS_REPORT.md`).

### Changed
- Extensive repository refactoring and cleanup; moved legacy components to `archive/`.
- Modified `src/dataset.py` to securely stream in-memory external dataset records alongside MTSamples natively.
- Stratified `config/dataset_config.yaml` to govern external data federation explicitly.

### Removed
- Cleaned up obsolete cache artifacts and deprecated preprocessing scripts.
- Removed legacy `print_cells.py` from `scratch/`.

### Verified
- Tested and stabilized inference paths, loss configurations, checkpoint management, and multi-process data loaders.
- Confirmed full test suite passage (161 tests) and reproducibility state locking (fixed random seed 1337).