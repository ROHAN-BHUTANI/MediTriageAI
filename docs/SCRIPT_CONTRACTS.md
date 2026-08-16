# MediTriageAI — Executable Script Input/Output Contracts

**Specification Baseline:** `v1.0.0-FROZEN`  
**Document Status:** AUTHORITATIVE SCRIPT INTERFACES  
**Date:** `2026-08-16`

---

## 1. Script Contract Specifications

### 1. `scripts/build_canonical.py`
- **Subsystem:** Canonical Dataset Build
- **Inputs:** 
  - `datasets/raw/mtsamples/mtsamples (1).csv`
  - `datasets/raw/neiss/neiss_all.parquet`
  - `datasets/raw/nhamcs_ed/ed2019/`, `ed2020/`, `ed2021/`
  - `datasets/raw/kaggle_medical_triage/data/*.parquet`
  - `datasets/raw/symptom2disease/Symptom2Disease.csv`
  - `meditriage/builder/canonical_schema.py`
- **Outputs:**
  - `meditriage/data/canonical/v1.0.0/dataset.parquet`
  - `meditriage/data/canonical/v1.0.0/build_manifest.json`
  - `meditriage/data/canonical/v1.0.0/dataset_gate_01.json`
- **CLI Options:** `--output-dir DIR`, `--seed INT` (default 42), `--peds-override`
- **Preconditions:** Raw datasets downloaded and verified against `DATASET_LICENSE_REGISTER.md`.
- **Postconditions:** 
  - 26-field canonical Parquet written with 0 schema errors.
  - Deterministic stratified 80/10/10 split.
  - 0 cross-split source_record_id or exact normalized text leakage.
  - DATASET-GATE-01 evaluated and written.
- **Failure Conditions:** Missing raw files, schema validation error, cross-split leakage, non-unique source IDs.
- **Safe to Run:** YES (Local CPU/RAM).

---

### 2. `scripts/build_pilot.py`
- **Subsystem:** Pilot Dataset Build
- **Inputs:** Same raw inputs as `build_canonical.py` (sampled).
- **Outputs:**
  - `meditriage/data/canonical/pilot/dataset.parquet`
  - `meditriage/data/canonical/pilot/build_manifest.json`
- **CLI Options:** `--output-dir DIR`, `--seed INT` (default 42), `--peds-override`
- **Preconditions:** Raw datasets present.
- **Postconditions:** 1,398-row pilot Parquet matching canonical schema.
- **Failure Conditions:** Missing raw files, schema errors.
- **Safe to Run:** YES (Local fast execution, <2 seconds).

---

### 3. `scripts/flight_check.py`
- **Subsystem:** Quality Gate & Pre-Training Flight Check
- **Inputs:**
  - `meditriage/data/canonical/v1.0.0/dataset.parquet`
  - `meditriage/data/canonical/v1.0.0/build_manifest.json`
  - `docs/specification/frozen/v1.0.0/**`
  - `tests/test_canonical_pipeline.py`
- **Outputs:**
  - Console flight check report with `PASS` / `FAIL` status for 18 flight items.
  - Exit code 0 (PASS) or 1 (FAIL).
- **CLI Options:** `--dataset-dir DIR`, `--skip-pytest`
- **Preconditions:** Canonical dataset built.
- **Postconditions:** Verified that dataset, schema, tests, model zoo, and frozen spec are 100% compliant.
- **Failure Conditions:** Any test failure, schema violation, leakage, hash mismatch, or modified frozen spec.
- **Safe to Run:** YES (Read-only verification, <5 seconds).

---

### 4. `scripts/train.py`
- **Subsystem:** Local Single-GPU Training Harness
- **Inputs:**
  - Canonical dataset Parquet (`meditriage/data/canonical/v1.0.0/dataset.parquet` or configured path)
  - Pretrained transformer weights (`xlm-roberta-base`, `google/muril-base-cased`, etc.)
  - Training configuration (`src.config_manager.TrainingConfig`)
- **Outputs:**
  - `results/{model_name}/checkpoint.pt`
  - `results/{model_name}/metrics.json`
  - `results/{model_name}/metadata.json`
- **CLI Options:** `--model-name`, `--batch-size`, `--epochs`, `--lr`, `--dataset-path`
- **Preconditions:** `flight_check.py` PASS; GPU available or CPU mode explicitly enabled.
- **Postconditions:** Checkpoint saved with loss and evaluation metrics recorded.
- **Failure Conditions:** OOM, dataset path mismatch, missing weights.
- **Safe to Run:** Authorized for development/local debugging.

---

### 5. `scripts/train_ddp.py`
- **Subsystem:** Multi-GPU Distributed Data Parallel Training (DGX)
- **Inputs:** Same as `scripts/train.py` + `torch.distributed` environment variables (`RANK`, `WORLD_SIZE`, `LOCAL_RANK`).
- **Outputs:** Same as `scripts/train.py` + distributed rank logs.
- **CLI Options:** `--config PATH`, `--mode {smoke,development,publication}` (Launch: `torchrun --nproc_per_node=N scripts/train_ddp.py --config configs/production_xlm_roberta.yaml --mode publication`)
- **Preconditions:** Multi-GPU environment; DATASET-GATE-01 PASS; dataset checksum matches.
- **Postconditions:** Synchronized DDP model checkpoint saved from rank 0.
- **Failure Conditions:** NCCL timeout, rank divergence, dataset checksum mismatch.
- **Safe to Run:** ONLY ON DGX / MULTI-GPU CLUSTER.

---

### 6. `scripts/evaluate.py`
- **Subsystem:** Model Evaluation & Statistical Verification
- **Inputs:**
  - Model checkpoint (`results/{model_name}/checkpoint.pt`)
  - Canonical dataset test split (`meditriage/data/canonical/v1.0.0/dataset.parquet`)
- **Outputs:**
  - `results/{model_name}/classification_report.txt`
  - `results/{model_name}/confusion_matrix.png`
  - `results/{model_name}/calibration.csv`
  - `results/{model_name}/evaluation_report.json`
- **CLI Options:** `--checkpoint PATH`, `--dataset PATH`, `--output-dir DIR`, `--bootstrap-resamples INT`
- **Preconditions:** Trained checkpoint exists and matches architecture config.
- **Postconditions:** Full metrics calculated (Macro-F1, AUROC, ECE, MAE, ordinal confusion) with bootstrap 95% CIs.
- **Failure Conditions:** Checkpoint mismatch, missing test split.
- **Safe to Run:** YES (Local CPU or single GPU).

---

### 7. `scripts/dataset_audit.py`
- **Subsystem:** Standalone Dataset Statistics & Class Entropy
- **Inputs:** Any dataset Parquet file.
- **Outputs:** `results/dataset_audit/audit_results.json`, `audit_summary.md`.
- **Preconditions:** Dataset exists.
- **Postconditions:** Class balance, missing value rates, and entropy metrics calculated.
- **Safe to Run:** YES.
