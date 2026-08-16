# MediTriageAI Data Engine & Modeling System (PROJECT.md)

**Repository:** `ROHAN-BHUTANI/MediTriageAI`  
**Current Release Version:** `1.0.0`  
**Current Phase / Status:** Phase 5 Completed (Frozen Baseline)

---

## 1. Executive Summary

MediTriageAI is a comprehensive, publication-ready research platform designed for multilingual clinical triage, specialty department routing, and severity acuity assessment. It integrates a robust data engineering engine (unifying 13+ medical corpora) with a multi-task dual-head Transformer architecture capable of joint learning across disjoint and partially-labeled clinical datasets.

---

## 2. Core Subsystems

1. **Data Engineering Engine (`meditriage/builder`, `reconstruction/`, `datasets/`)**:
   - Ingests, cleans, deduplicates, and validates patient presentations across 13 diverse open clinical datasets.
   - Enforces a unified 5-point severity taxonomy (`S1` to `S5`) and 13-class specialist department routing taxonomy.
   - Includes a 10-stage dataset reconstruction pipeline with synthetic generation and deficit augmentation.

2. **Model Zoo & Multi-Task Training (`models/`, `meditriage/training/`, `src/`)**:
   - **Model Zoo Options:** XLM-RoBERTa Large, mBERT, DistilBERT Multilingual, IndicBERT, and Emergent Path Triage (`E-PATH-CO-REASON`).
   - **Joint Masked Focal Loss:** Enables gradient updates on dual, single, or disjoint labels with loss weighting ($\alpha=1.0, \beta=1.2$).
   - **Hardware Harnesses:** Colab T4, Single-GPU, and Multi-GPU DDP (`scripts/train_ddp.py`) with Rank-0 safe file operations.

3. **Evaluation, Verification & Serving (`analysis/`, `meditriage/evaluation/`, `scripts/serve_api.py`)**:
   - 95% Bootstrap Confidence Intervals (1,000 resamples), ECE/MCE calibration, Cohen's Kappa, and McNemar tests.
   - FastAPI REST endpoint with rule-based emergency red-flag fallbacks.
   - Interactive local browser dashboard (`dashboard_web/`).

---

## 3. Key Constraints & Non-Functional Requirements

- **Research Disclaimer:** Not for live clinical diagnostic or triage use without formal clinical safety audits.
- **Determinism:** Seeded random execution (`1337`), deterministic CuDNN, and dataset SHA256 checksum manifests.
- **Test Coverage:** 444 automated unit and integration tests passing across all components.
