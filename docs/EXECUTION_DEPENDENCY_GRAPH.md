# MediTriageAI — End-to-End Execution Dependency Graph

**Specification Baseline:** `v1.0.0-FROZEN`  
**Document Status:** AUTHORITATIVE EXECUTION FLOW  
**Date:** `2026-08-16`

---

## 1. End-to-End System Pipeline Graph

```mermaid
flowchart TD
    subgraph S1 [Stage 1: Raw Acquisition & Governance]
        R1[datasets/raw/mtsamples/]
        R2[datasets/raw/neiss/neiss_all.parquet]
        R3[datasets/raw/nhamcs_ed/]
        R4[datasets/raw/kaggle_medical_triage/]
        R5[datasets/raw/symptom2disease/]
        LIC[docs/specification/dataset/DATASET_LICENSE_REGISTER.md]
    end

    subgraph S2 [Stage 2: Canonical Dataset Build Pipeline]
        BC[scripts/build_canonical.py]
        SCHEMA[meditriage/builder/canonical_schema.py]
        VARIATION[meditriage/multilingual/variation/generators.py]
        OFFLINE[meditriage/multilingual/providers/offline.py]
        
        INGEST[1. Raw Ingestion & Source Normalization]
        QC[2. Quality Control & CJK Safety Filter]
        DEDUP[3. Deduplication & Global ID Assignment]
        SPLIT[4. Source-Aware Stratified 80/10/10 Split]
        AUG[5. Multi-Strata Linguistic & Robustness Augmentation]
        LINEAGE[6. Augmentation Lineage & Cross-Split Leak Check]
        EXPORT[7. Parquet & Manifest Export]
    end

    subgraph S3 [Stage 3: Output Artifacts & Gate Validation]
        PQ[(meditriage/data/canonical/v1.0.0/dataset.parquet)]
        MAN[meditriage/data/canonical/v1.0.0/build_manifest.json]
        GATE_JSON[meditriage/data/canonical/v1.0.0/dataset_gate_01.json]
        GATE_REP[docs/specification/audits/DATASET_GATE_01_REPORT.md]
        FC[scripts/flight_check.py]
        TESTS[tests/test_canonical_pipeline.py]
    end

    subgraph S4 [Stage 4: Training Harness]
        TRAIN_CLI[scripts/train.py / scripts/train_ddp.py]
        DATA_LOADER[src/data_pipeline.py]
        MODEL[src/model.py MediTriageTransformer]
        TRAINER[src/trainer.py Masked Focal Loss]
        CKPT[(checkpoints/best_model.pt)]
    end

    subgraph S5 [Stage 5: Statistical Evaluation & Release]
        EVAL[scripts/evaluate.py]
        METRICS[src/metrics.py Macro-F1 / ECE / MAE]
        EXPORTER[src/evaluation.py EvaluationExporter]
        EVAL_JSON[(results/evaluation_report.json)]
        DASH[src/dashboard.py Streamlit Research Demo]
    end

    %% Linkages
    R1 & R2 & R3 & R4 & R5 & LIC --> INGEST
    INGEST --> QC --> DEDUP --> SPLIT --> AUG --> LINEAGE --> EXPORT
    SCHEMA & VARIATION & OFFLINE -.-> BC
    BC --> EXPORT
    EXPORT --> PQ & MAN & GATE_JSON & GATE_REP
    
    PQ & MAN --> FC & TESTS
    FC & TESTS -- "GATE 01 PASS" --> TRAIN_CLI
    
    PQ --> DATA_LOADER
    TRAIN_CLI --> DATA_LOADER --> MODEL --> TRAINER --> CKPT
    
    CKPT & PQ --> EVAL
    EVAL --> METRICS --> EXPORTER --> EVAL_JSON
    EVAL_JSON & CKPT --> DASH
```

---

## 2. Linear Execution Stage Sequence

Every official benchmark and training campaign strictly follows this 15-stage sequence:

| Step | Stage Name | Governing Script / Module | Input Artifact(s) | Output Artifact(s) | Blocking Gate |
|---|---|---|---|---|---|
| **01** | `INPUT` | `datasets/raw/**` | External download archives | Raw local files | License clearance |
| **02** | `INGESTION` | `scripts/build_canonical.py` | Raw CSV/Parquet/FWF | Raw record dicts | Source existence |
| **03** | `NORMALIZATION` | `scripts/build_canonical.py` | Raw record dicts | Standard text & headers | No empty strings |
| **04** | `SCHEMA` | `canonical_schema.py` | Standardized dicts | Pre-validated records | 26-field conformance |
| **05** | `LANGUAGE` | `detect_script()` | Text content | Language & script tags | CJK quarantine |
| **06** | `DEDUP` | `deduplicate()` | Pre-split records | Unique records | 0 exact duplicate texts |
| **07** | `SPLIT` | `assign_stratified_splits()`| Deduplicated records | Split-assigned records (80/10/10) | Source & group isolation |
| **08** | `AUGMENTATION` | `augment_records()` | Source records with split | Augmented records (Strata 1–10) | Semantic preservation |
| **09** | `VALIDATION` | `check_leakage()`, `validate_records()` | Full merged dataset | Validation report dict | 0 schema errors, 0 cross-split leaks |
| **10** | `MANIFEST` | `build_manifest.json` builder | Dataset Parquet table | `dataset.parquet` + `build_manifest.json` | SHA-256 computation |
| **11** | `DATASET-GATE-01`| `evaluate_dataset_gate_01()`| Manifest & Parquet | `dataset_gate_01.json`, `DATASET_GATE_01_REPORT.md`| **DATASET-GATE-01 PASS** |
| **12** | `TRAINING` | `scripts/train.py` / `train_ddp.py` | `dataset.parquet` + Config | Checkpoint weights & tensorboard logs | Masked loss convergence |
| **13** | `CHECKPOINT` | `src/checkpoint_manager.py` | Trained weights | `best_model.pt` + `metadata.json` | Checkpoint hash |
| **14** | `EVALUATION` | `scripts/evaluate.py` | Checkpoint + Test split | `evaluation_report.json` | Macro-F1 & ECE computation |
| **15** | `RELEASE` | `scripts/paper/manifest.py` | Evaluation JSON + Model | Final paper tables & figures | Byte-identical disclaimer |
