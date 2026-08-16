# MediTriageAI — Authoritative Subsystem Entrypoints

**Specification Baseline:** `v1.0.0-FROZEN`  
**Document Status:** AUTHORITATIVE ENTRYPOINT REGISTRY  
**Date:** `2026-08-16`

---

## 1. Primary Authoritative Entrypoints

To prevent execution ambiguity, each major subsystem has exactly ONE authoritative entrypoint.

| Subsystem Responsibility | Primary Authoritative Entrypoint | Classification | Direct Inbound Dependencies | Expected Execution Context |
|---|---|---|---|---|
| **Canonical Dataset Build** | `scripts/build_canonical.py` | **PRIMARY** | `meditriage.builder`, `datasets/raw/**` | Local / Builder Environment |
| **Pilot Dataset Build** | `scripts/build_pilot.py` | **PRIMARY PILOT** | `meditriage.builder`, `datasets/raw/**` | Local Verification |
| **Pre-Training Flight Check** | `scripts/flight_check.py` | **PRIMARY GATE** | `meditriage/data/canonical/v1.0.0/` | Local / CI / Pre-DGX |
| **Single-GPU Training** | `scripts/train.py` | **PRIMARY TRAIN** | `src.trainer`, `src.model`, `src.data_pipeline` | Local Workstation / Single GPU |
| **Distributed Multi-GPU (DDP)** | `scripts/train_ddp.py` | **PRIMARY DDP** | `src.trainer`, `torch.distributed` | DGX / Cluster Environment |
| **Statistical Evaluation** | `scripts/evaluate.py` | **PRIMARY EVAL** | `src.evaluation`, `src.metrics` | Local / Post-Training |
| **Robustness & Model Zoo Test** | `tests/test_model_zoo.py` | **PRIMARY ZOO** | `models/`, `src/model.py` | Pytest Harness |
| **Research Demo / UI** | `src/dashboard.py` | **PRIMARY DEMO** | `streamlit`, `src.model` | Interactive Demo |
| **Publication Artifacts** | `scripts/paper/manifest.py` | **PRIMARY PAPER** | `results/**`, `scripts/paper/` | Publication Generator |

---

## 2. Secondary, Experimental, and Legacy Entrypoints

| Script | Classification | Current Role | Migration Recommendation |
|---|---|---|---|
| `scripts/colab_train.py` | **SECONDARY** | Lightweight Google Colab single-GPU training helper | Retain as secondary cloud wrapper |
| `scripts/dataset_audit.py` | **SECONDARY** | Standalone dataset class entropy analysis | Retain as diagnostic utility |
| `scripts/error_analysis.py` | **SECONDARY** | Post-evaluation error slicing tool | Retain as evaluation utility |
| `scripts/multilingual_expansion.py` | **EXPERIMENTAL** | Standalone multilingual augmentation pipeline | Subsumed into `build_canonical.py` stage 14 |
| `scripts/phenotype_augmentation.py` | **EXPERIMENTAL** | Clinical phenotype expansion generator | Subsumed into `build_canonical.py` / research |
| `scripts/hard_negative_generation.py` | **EXPERIMENTAL** | Standalone hard negative generator | Subsumed into `build_canonical.py` stage 14 |
| `scripts/dry_run.py` | **LEGACY** | Historical dry run test of multi-stage builder | Deprecated; superseded by `build_pilot.py` |
| `scripts/dataset_quality_improvement.py`| **LEGACY** | Historical dataset filtering experiment | Deprecated; superseded by `build_canonical.py` |
| `reconstruction/run.py` | **ARCHIVE** | Historical 10-stage dataset reconstruction pipeline | Preserve as historical reference; do not execute |
