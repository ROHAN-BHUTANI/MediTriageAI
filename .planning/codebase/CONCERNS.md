# MediTriageAI — Known Concerns, Limitations & Technical Debt (CONCERNS.md)

**Generated:** 2026-08-14  
**Repository State:** Frozen Baseline (v1.0.0)

---

## 1. Technical Debt & Codebase Organization

1. **Parallel Subsystems (`src/` vs `meditriage/`)**:
   - `src/` represents the foundational research / Colab training codebase used during earlier exploration and paper prototyping.
   - `meditriage/` is the modular, production-ready refactor with isolated subpackages (`builder/`, `training/`, `evaluation/`, `multilingual/`).
   - *Impact:* Future extensions should standardize on `meditriage/` to eliminate dual-maintenance across `src/trainer.py` and `meditriage/training/trainer.py`.

2. **Pending Code TODOs**:
   - Found at `src/evaluation.py:137`:
     ```python
     "config_hash": "TODO_config_hash",
     ```
   - In production runs, config hashes are properly populated via `Config.get_hash()`, but this placeholder remains in the legacy evaluation export routine.

---

## 2. Modeling & Clinical Constraints

1. **Clinical Non-Validation (Ethics Disclaimer)**:
   - This framework is a **research prototype** and must NOT be used for direct real-world patient triage or medical decision-making without formal clinical safety audits and regulatory clearance.
2. **Sequence Truncation**:
   - Transformer encoders are constrained to `max_length=512` (and `max_length=128` on Colab T4). Long clinical narratives, multi-day consultation transcripts, and extended hospital histories are truncated, potentially omitting downstream discharge notes or delayed symptoms.
3. **Specialist Group Granularity**:
   - 13 canonical department buckets (`CARDIO_PULM`, `SURGERY`, `ED`, etc.) were consolidated to handle taxonomic overlap across disparate datasets. Highly specific subspecialties (e.g., pediatric nephrology, neuro-oncology) are lumped into broader groups.

---

## 3. Infrastructure & Scaling

1. **Distributed Scaling Limits**:
   - PyTorch DistributedDataParallel (DDP) is fully tested for single-node multi-GPU (e.g., 4x or 8x A100/V100). Multi-node scaling using DeepSpeed ZeRO-3 or PyTorch Fully Sharded Data Parallel (FSDP) has not yet been integrated.
2. **Local Storage & Datasets Bundle**:
   - The workspace contains a ~1.0 GB binary tarball `datasets_bundle.tar.gz` and multiple multi-megabyte CSV files in `data/`. Ensure `.gitignore` and LFS policies prevent accidental bloat in Git history.
