# MediTriageAI — Master Flight Checklist

**Specification Baseline:** `v1.0.0-FROZEN`  
**Document Status:** AUTHORITATIVE VERIFICATION CHECKLIST  
**Date:** `2026-08-16`

---

## 1. Current Flight Status Matrix

| Stage | Checklist Item | Status | Precondition | Command | Expected Output | Pass Condition |
|---|---|---|---|---|---|---|
| **0.0** | Specification Immutability | `[x] PASS` | Git clean | `git diff docs/specification/frozen/` | Clean diff | 0 bytes modified in frozen spec |
| **0.1** | Historical Dataset Preserved | `[x] PASS` | Baseline exists | `Get-Item meditriage/data/processed/dataset.parquet` | 1,306,576,608 bytes | File intact and unmodified |
| **1.0** | Raw Datasets Available | `[x] PASS` | Downloaded | `ls datasets/raw/` | 5 source directories | 5 Grade-A sources present |
| **1.1** | License Gate Cleared | `[x] PASS` | License register | `DATASET_LICENSE_REGISTER.md` | 5 Grade-A, 0 D/E | 0 rejected sources ingested |
| **2.0** | Canonical Dataset Build | `[x] PASS` | Raw sources | `python scripts/build_canonical.py` | 53,067 records | Parquet, manifest, gate JSON written |
| **2.1** | Byte-for-Byte Reproducibility | `[x] PASS` | Fixed seed 42 | Independent 2nd build test | Identical SHA-256 | `f64ed360b246416c...` exact match |
| **3.0** | Canonical Schema Validity | `[x] PASS` | Dataset built | `python scripts/flight_check.py` | 0 schema errors | 26/26 fields valid |
| **3.1** | Zero Leakage Audit | `[x] PASS` | Dataset built | `python scripts/flight_check.py` | 0 cross-split IDs/texts | 0 train/val/test crossings |
| **4.0** | DATASET-GATE-01 Evaluation | `[x] PASS` | Gate report | `DATASET_GATE_01_REPORT.md` | 18 requirements met | `overall: PASS` |
| **4.1** | Test Suite Verification | `[x] PASS` | Pytest installed | `pytest tests/test_canonical_pipeline.py` | 58 passed | 58/58 tests passing |
| **5.0** | Model-Dataset Alignment | `[x] PASS` | Codebase mapped | `src/model.py`, `src/specialty_mapping.py`| 13 depts, 5 ESI | Logits match 1-to-1 |
| **6.0** | DGX Training Pre-Flight | `[x] PASS` | All gates pass | `python scripts/flight_check.py` | `OVERALL: PASS` | Pre-flight check exit code 0 |
| **7.0** | Model Training Execution | `[ ] NOT STARTED` | Authorization | `scripts/train.py` / `train_ddp.py` | Trained weights | Awaiting separate authorization |
| **8.0** | Statistical Evaluation | `[ ] NOT STARTED` | Checkpoint saved | `scripts/evaluate.py` | Evaluation report | Macro-F1 / ECE / MAE generated |
| **9.0** | Subgroup Robustness Audit | `[ ] NOT STARTED` | Evaluation run | `scripts/error_analysis.py` | Error breakdown | Robustness metrics logged |
| **10.0**| Code Governance & Release | `[ ] NOT STARTED` | Results ready | `scripts/paper/manifest.py` | Release envelope | Release manifest signed |

---

## 2. Stage-by-Stage Verification Contracts

### Stage 2: Canonical Build
- **PRECONDITION:** 5 Grade-A raw sources present in `datasets/raw/`.
- **COMMAND:** `python scripts/build_canonical.py --output-dir meditriage/data/canonical/v1.0.0/ --seed 42`
- **EXPECTED OUTPUT:** `FULL BUILD: PASS` with 53,067 records.
- **PASS CONDITION:** SHA-256 matches `f64ed360b246416cf3b117a27f9c09843f1ad53430a3fd2575358587c1902513`.
- **FAIL CONDITION:** Any schema errors, non-unique source IDs, or cross-split leakage.
- **ARTIFACT:** `meditriage/data/canonical/v1.0.0/dataset.parquet`

### Stage 3: Flight Check
- **PRECONDITION:** Canonical dataset Parquet and manifest exist.
- **COMMAND:** `python scripts/flight_check.py`
- **EXPECTED OUTPUT:** `OVERALL STATUS: [PASS] ALL FLIGHT CHECKS PASSED — READY FOR TRAINING`.
- **PASS CONDITION:** All 14 automated check items return `[PASS]`.
- **FAIL CONDITION:** Any item returns `[FAIL]`.
- **ARTIFACT:** Console flight check report.

### Stage 4: Gate 01 Audit
- **PRECONDITION:** Build completed.
- **COMMAND:** Inspect `meditriage/data/canonical/v1.0.0/dataset_gate_01.json`.
- **EXPECTED OUTPUT:** `overall: PASS`, `binding_failures: []`.
- **PASS CONDITION:** 18/18 requirements satisfied.
- **ARTIFACT:** `docs/specification/audits/DATASET_GATE_01_REPORT.md`.
