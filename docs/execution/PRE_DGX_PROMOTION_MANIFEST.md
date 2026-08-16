# MediTriageAI — Pre-DGX Promotion Manifest

**Promotion Date:** `2026-08-16`  
**Specification Baseline:** `v1.0.0-FROZEN`  
**Promotion Authorization:** **AUTHORIZED FOR DGX TRAINING**

---

## 1. Provenance & Artifact Signatures

| Metadata Field | Canonical Value |
|---|---|
| **Git Branch** | `training-pipeline` |
| **Commit Baseline** | `4b5f4a0417fb492c1460a028a71902e00757148e` |
| **Promotion Commit** | To be determined upon commit & push |
| **Remote Repository** | `https://github.com/ROHAN-BHUTANI/MediTriageAI.git` |
| **Canonical Dataset Path** | `meditriage/data/canonical/v1.0.0/dataset.parquet` |
| **Dataset SHA-256 Checksum** | `f64ed360b246416cf3b117a27f9c09843f1ad53430a3fd2575358587c1902513` |
| **Dataset Total Rows** | `53,067` |
| **Dataset Splits** | Train: `42,414` (79.9%), Val: `5,298` (10.0%), Test: `5,355` (10.1%) |
| **Model Identifier** | `xlm-roberta-base` (278,057,490 params) |
| **Tokenizer Identifier** | `xlm-roberta-base` (SentencePiece BPE 250k vocab) |
| **Specialist Routing Head** | `nn.Linear(768, 13)` (13 clinical departments) |
| **Severity Acuity Head** | `nn.Linear(768, 5)` (ESI levels S1–S5, missing masked via `ignore_index=-1`) |
| **Joint Loss Weights** | $\alpha = 1.0, \beta = 1.2, \gamma = 2.0$ |
| **Python Environment** | Python 3.12.10 |
| **PyTorch / Transformers** | PyTorch 2.6.0, Transformers 4.49.0 |
| **CUDA Expectation** | CUDA 12.1+ (NVIDIA Driver 535+) |

---

## 2. Gate & Verification Summary

| Gate / Audit Layer | Evaluator | Result | Key Details |
|---|---|---|---|
| **DATASET-GATE-01** | `evaluate_dataset_gate_01()` | **PASS** | 18 requirements met; 0 binding failures |
| **Automated Flight Check** | `scripts/flight_check.py` | **PASS** | 14/14 automated flight checks passed |
| **Canonical Pipeline Tests** | `pytest tests/test_canonical_pipeline.py` | **PASS** | 58/58 pytest tests passed in 6.43s |
| **Training Smoke Test** | `scratch/run_smoke_test.py` | **PASS** | Complete forward, backward, checkpoint, and evaluation cycle succeeded (Exit code 0) |
| **Reproducibility Test** | Multi-process build validation | **PASS** | Byte-for-byte SHA-256 match confirmed across independent processes |
| **Frozen Spec Immutability** | `git diff docs/specification/frozen/` | **PASS** | 0 bytes modified |
| **GSD Inspection** | Structural & plan verification | **PASS** | Training-ready implementation strictly adheres to specification |
| **Ralph Bounded Cleanup** | Engineering cleanup & validation | **PASS** | Path portability & dataset path alignments verified |
| **CodeRabbit Static Review**| Code diff analysis | **PASS** | 0 blocking / high issues |

---

## 3. Monitored Non-Blocking Risks (Post-Training Evaluation Required)

1. **English Dominance:** ~98.1% of dataset records are English. Multilingual generalization will be evaluated post-training on `hi-en`, `hi`, and `hi-Latn` test splits.
2. **Rare Clinical Classes:** `OBGYN` (80 train rows), `PEDS` (150 train rows), `ONCOLOGY_HEME` (178 train rows), `PSYCH` (312 train rows) and severity levels `S1` (261 train rows), `S5` (616 train rows).
3. **Evaluation Mandate:** Subgroup Macro-F1 metrics across all 13 departments, 5 severity levels, and 6 robustness strata must be published in `results/evaluation_report.json`.

---

## 4. Promotion Authorization

============================================================
**PROMOTION STATUS: AUTHORIZED FOR DGX TRAINING**
============================================================
