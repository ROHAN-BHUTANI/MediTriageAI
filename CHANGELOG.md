# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-07-30

### Added
- **Phase 1: Dataset Generation & Portability** 
  - Complete ingestion, cleaning, normalization and deductive pipelines.
  - Integration of 13+ medical dataset sources (e.g., NHAMCS, MTSamples, MedQA, etc.).
  - Cross-platform CLI export capability (JSON, CSV, Parquet) replacing dependency on `.sixth`.
- **Phase 2: Configuration & Initialization**
  - Reproducible `TrainingConfig` YAML/Dataclass hybrid system.
  - Manifest validation for ensuring identical dataset states.
  - Deterministic random seeding and process stability checks.
- **Phase 3: Multi-Task Losses & Validation**
  - Dynamic loss masking to handle missing department or severity labels across disjoint datasets.
  - Multi-head inference evaluation with Macro, Micro, and Weighted F1 metrics alongside AUROC and confusion matrices.
- **Phase 4: Output, Evaluation & Reporting**
  - Integrated export of prediction statistics (`predictions.parquet`, `correct.csv`, `misclassified.csv`, `entropy_distribution.csv`).
  - Automated reporting generation logic.
- **Phase 5: DGX Scalability & Production Hardening**
  - `train_ddp.py` entrypoint for NCCL DistributedDataParallel scaling.
  - Added support for `torch.compile` (`use_torch_compile`), automatic mixed precision (AMP), gradient checkpointing, and Flash Attention.
  - Hardened multi-node failure handling (process group teardown).
  - Multi-GPU communication efficiency enhancements.

### Changed
- Refactored `run_experiment.py` into distinct components and modularized evaluation into `evaluate.py`.
- Normalized triage labels across datasets strictly to integers (1-5).
- Re-architected output representations, completely deprecating nested objects for flattened tensor tuples (`spec_logits`, `sev_logits`) to avoid Python object serialization crashes in distributed environments.

### Fixed
- Addressed label leakage in train/val datasets and deduplicated cross-splits.
- Fixed DDP object instantiation bugs regarding config injection missing parameters (`use_torch_compile`, `non_blocking_transfers`).
- Resolved string normalization coercion issues for NHAMCS datasets.
