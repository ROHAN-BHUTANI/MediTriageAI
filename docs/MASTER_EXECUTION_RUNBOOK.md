# MediTriageAI — Master Execution Runbook

**Specification Baseline:** `v1.0.0-FROZEN`  
**Document Status:** AUTHORITATIVE RUNBOOK FOR ENGINEERS & AI AGENTS  
**Date:** `2026-08-16`

---

## Stage 0: Environment Verification

1. **Repository Checkout & State Verification:**
   ```bash
   git status
   git diff --name-only
   ```
   *Pass Condition:* No modifications to `docs/specification/frozen/v1.0.0/**` or `meditriage/data/processed/dataset.parquet`.

2. **Python Environment & Dependencies:**
   ```bash
   python --version  # Python 3.10+
   python -c "import torch, transformers, pyarrow, pandas, pytest; print('Environment: PASS')"
   ```

3. **GPU / CUDA Availability (for training/evaluation stages):**
   ```bash
   python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"
   ```

---

## Stage 1: Data Acquisition & License Verification

1. **Verify Raw Datasets:**
   Check that all 5 Grade-A open datasets exist locally:
   - `datasets/raw/mtsamples/`
   - `datasets/raw/neiss/neiss_all.parquet`
   - `datasets/raw/nhamcs_ed/`
   - `datasets/raw/kaggle_medical_triage/`
   - `datasets/raw/symptom2disease/`

2. **Verify License Register:**
   Review `docs/specification/dataset/DATASET_LICENSE_REGISTER.md` to ensure no Grade D/E sources are ingested.

---

## Stage 2: Canonical Dataset Construction

Execute the authoritative 20-stage build pipeline:
```bash
python scripts/build_canonical.py --output-dir meditriage/data/canonical/v1.0.0/ --seed 42
```
*Expected Outputs:*
- `meditriage/data/canonical/v1.0.0/dataset.parquet` (53,067 records)
- `meditriage/data/canonical/v1.0.0/build_manifest.json` (SHA-256: `f64ed360b246416cf3b117a27f9c09843f1ad53430a3fd2575358587c1902513`)
- `meditriage/data/canonical/v1.0.0/dataset_gate_01.json` (`overall: PASS`)

---

## Stage 3: Pre-Training Flight Check

Run the automated flight check script:
```bash
python scripts/flight_check.py
```
*Pass Condition:* `OVERALL STATUS: [PASS] ALL FLIGHT CHECKS PASSED — READY FOR TRAINING`.

---

## Stage 4: DATASET-GATE-01 Confirmation

Confirm that `docs/specification/audits/DATASET_GATE_01_REPORT.md` is populated and all 18 requirements are satisfied.

---

## Stage 5: Training Preparation & Configuration

1. **Configuration Validation:**
   Confirm hyperparameters in `src/config_manager.py`:
   - `alpha_specialist = 1.0`
   - `beta_severity = 1.2`
   - `gamma = 2.0` (Focal Loss)
   - `ignore_index = -1` (Masking missing severity labels)
   - `learning_rate = 2e-5`
   - `batch_size = 32` (or hardware-scaled)
   - `max_length = 512`

2. **Checkpoint Directory Setup:**
   ```bash
   mkdir -p results/xlm_roberta_large
   mkdir -p results/indic_bert
   ```

---

## Stage 6: DGX Multi-GPU Cluster Launch

On multi-GPU DGX nodes (8x NVIDIA H100 / A100):
```bash
torchrun --nproc_per_node=8 scripts/train_ddp.py \
    --model-name xlm-roberta-base \
    --dataset-path meditriage/data/canonical/v1.0.0/dataset.parquet \
    --output-dir results/xlm_roberta_large \
    --batch-size 32 \
    --epochs 10 \
    --lr 2e-5
```

---

## Stage 7: Local Single-GPU Training

For single-GPU execution or verification runs:
```bash
python scripts/train.py \
    --model-name xlm-roberta-base \
    --dataset-path meditriage/data/canonical/v1.0.0/dataset.parquet \
    --output-dir results/xlm_roberta_large \
    --batch-size 16 \
    --epochs 5
```

---

## Stage 8: Evaluation & Statistical Verification

Run comprehensive evaluation on the test split:
```bash
python scripts/evaluate.py \
    --checkpoint results/xlm_roberta_large/checkpoint.pt \
    --dataset meditriage/data/canonical/v1.0.0/dataset.parquet \
    --output-dir results/xlm_roberta_large \
    --bootstrap-resamples 1000
```
*Outputs Generated:*
- `results/xlm_roberta_large/classification_report.txt`
- `results/xlm_roberta_large/confusion_matrix.png`
- `results/xlm_roberta_large/calibration.csv`
- `results/xlm_roberta_large/evaluation_report.json`

---

## Stage 9: Robustness & Subgroup Analysis

Run subgroup and error slicing across robustness dimensions:
```bash
python scripts/error_analysis.py \
    --results-dir results/xlm_roberta_large
```

---

## Stage 10: Automated Review & Code Governance

Execute post-build code review tools (CodeRabbit / Ralph / GSD):
- **Constraint:** These tools may review implementation code in `scripts/`, `src/`, and `meditriage/`, but are **STRICTLY FORBIDDEN** from modifying `docs/specification/frozen/v1.0.0/**`.

---

## Stage 11: Release Artifact Generation

Generate publication-ready tables and figures:
```bash
python scripts/paper/tables.py
python scripts/paper/plots.py
python scripts/paper/manifest.py
```
