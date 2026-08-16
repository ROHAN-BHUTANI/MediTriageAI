# MediTriageAI — Testing Infrastructure & Verification (TESTING.md)

**Generated:** 2026-08-14  
**Repository State:** Frozen Baseline (v1.0.0)

---

## 1. Test Suite Overview

MediTriageAI features a comprehensive automated test suite consisting of **444 tests** across **40 test modules** located in `tests/` and `tests/builder/`.

| Test Category | Primary Files | Total Tests (Approx.) | Scope |
|:---|:---|:---:|:---|
| **Builder & Adapters** | `tests/builder/test_*.py` | 50+ | Ingestion adapters, governance, schema validation, stages, JSON/CSV exports |
| **Emergent Path Triage** | `tests/test_emergent_path_triage.py`, `test_dccf.py`, `test_amco.py`, `test_aces.py` | 100+ | Neural logic modules, dynamic routing, tensor shapes, weight initializations |
| **Training Framework** | `tests/test_training_framework.py`, `test_focal_loss.py`, `test_checkpoint_manager.py` | 30+ | Loss formulations, optimizer/scheduler factories, early stopping, RNG state resume |
| **Statistical & Evaluation** | `tests/test_research_validation.py`, `test_metrics.py`, `test_eval_loss_propagation.py` | 40+ | Bootstrap CIs, McNemar significance tests, ECE calibration, LaTeX exports |
| **Reconstruction Engine** | `tests/test_reconstruction.py`, `test_reconstruction_stages6_10.py` | 60+ | 10-stage dataset reconstruction, LLM providers, deficit augmentations, validators |
| **Research Framework (REF)** | `tests/test_ref_*.py` | 35+ | Benchmarks, metrics aggregation, provenance tracking, figure isolation |
| **Multilingual & Augmentation** | `tests/test_multilingual_expansion.py`, `test_clinical_*.py`, `test_llm_providers.py` | 45+ | Translation validation, hard negative generation, linguistic variation |
| **Schema & Data Integrity** | `tests/test_schema.py`, `test_schema_validation.py`, `test_sampling.py` | 20+ | Column renaming, unannotated rows, stratified sampling, dual-labeling |

---

## 2. Configuration & Execution

The test configuration is controlled by `pytest.ini` and `tests/conftest.py`:

```ini
[pytest]
pythonpath = .
```

### Running the Test Suite
- **Standard Fast Suite** (skips expensive full-epoch training tests):
  ```bash
  pytest
  ```
- **Full Suite including Slow Training Tests**:
  ```bash
  pytest --run-slow
  ```
- **Targeted Subsystem Test**:
  ```bash
  pytest tests/builder/
  pytest tests/test_training_framework.py
  pytest tests/test_emergent_path_triage.py
  ```

---

## 3. Mocking & Fixture Strategy

1. **Lightweight Tokenizer Fallback**: Unit tests avoid hitting external Hugging Face network endpoints by leveraging `SimpleClinicalTokenizer` or mocked token batches.
2. **CPU Mock Tensors**: Tests execute on CPU by default with small batch dimensions ($B=2, L=16, d=64$) for sub-second forward/backward verification.
3. **`tmp_path` Fixtures**: All file I/O tests utilize pytest's isolated `tmp_path` fixture to ensure no residual artifacts pollute the workspace.
4. **CUDA Fallback Shielding**: Tests verifying CUDA-specific routines (e.g. `test_cuda_rng_checkpoint_restoration`) safely test the branch logic even in non-CUDA test environments.
