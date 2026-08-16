# MediTriageAI — Repository Structure & Directory Map (STRUCTURE.md)

**Generated:** 2026-08-14  
**Repository State:** Frozen Baseline (v1.0.0)

---

## 1. Top-Level Directory Layout

```text
MediTriageAI_Data_Engine/
├── analysis/               # Statistical evaluation, calibration (ECE), bootstrapping, 300-DPI reporting
├── config/                 # Production training, dataset, enrichment, and reconstruction YAML configs
├── configs/                # Dataset audit and profiling configurations
├── dashboard_web/          # Interactive web UI (HTML5, JS, CSS) for triage exploration
├── data/                   # Cleaned, hinglish, and OOD benchmark datasets (CSV & Parquet)
├── datasets/               # HuggingFace download scripts, inventory generator, and dataset licenses
├── docs/                   # Governance policies, label provenance, schemas, and draft paper results
├── meditriage/             # Production modular package (builder, training, evaluation, multilingual)
├── models/                 # Model zoo implementations (XLM-R, mBERT, DistilBERT, IndicBERT, E-PATH)
├── paper_artifacts/        # Publication figures, LaTeX tables, manifests, and diagrams
├── reconstruction/         # 10-stage dataset reconstruction and synthetic generation engine
├── ref/                    # Research Experiment Framework (benchmarks, metrics, provenance, visualization)
├── scripts/                # Operational CLI entrypoints (train, train_ddp, evaluate, serve_api, etc.)
├── src/                    # Foundational core research engine and Colab training harness
├── tests/                  # Pytest test suite (444 tests across 40 test modules)
├── .planning/              # GSD specification, codebase mapping, and project state
├── pytest.ini              # Pytest configuration
├── requirements.txt        # Core Python dependencies
├── environment.yml         # Conda environment definition
├── VERSION                 # Current stable release version (1.0.0)
└── PROJECT_STATUS.md       # Status declaration (FROZEN)
```

---

## 2. Package Breakdown

### `meditriage/` (Modular Production Package)
- **`builder/`**:
  - `orchestrator.py`: Orchestrates multi-adapter ingestion and normalization.
  - `schema.py`: Pydantic / dataclass schema definitions.
  - `config.py`: Ingestion configuration loader and hash generator.
  - `adapters/`: 13 distinct dataset source adapters (`mtsamples.py`, `pmc_patients.py`, `chatdoctor_*.py`, etc.).
  - `stages/`: Ingestion stages (`normalize.py`, `deduplicate.py`, `augment.py`, `split.py`, `validate.py`).
- **`training/`**:
  - `trainer.py`: Production `MultiTaskClinicalClassifier` and `MultiTaskTrainer`.
  - `losses.py`: `MultiTaskLoss`, `FocalLoss`, and `WeightedCrossEntropyLoss`.
  - `callbacks.py` / `checkpoint.py`: Early stopping, metric tracking, and atomic checkpoint saving.
  - `metrics.py`: Macro-F1, Weighted-F1, per-class sensitivity, and specificity.
- **`evaluation/`**:
  - `benchmark_suite.py`: Multi-model comparative benchmark harness.
  - `significance.py`: Bootstrap confidence intervals and paired McNemar tests.
  - `latex_exporter.py`: Automated LaTeX table compilation for publication.
  - `robustness.py`: Linguistic noise, typographical error, and Hinglish robustness testing.
- **`multilingual/`**:
  - `translator.py` / `validator.py`: Cross-lingual medical translation and semantic preservation validator.
  - `providers/`: Modular LLM / translation API backends.

---

### `models/` (Model Zoo & Emergent Reasoning)
- `base_model.py`: Abstract `BaseMedicalTriageModel` and fallback `SimpleClinicalTokenizer`.
- `xlm_roberta.py`: Choice 1 (`XLMRobertaLargeModel`).
- `mbert.py`: Choice 2 (`MBertModel`).
- `distilbert_multi.py`: Choice 3 (`DistilBertMultilingualModel`).
- `indic_bert.py`: Choice 4 (`IndicBertModel`).
- `emergent_path_triage/`: Choice 5 (`EmergentPathTriageModel` / E-PATH-CO-REASON).
  - `dccf.py`, `amco.py`, `dces.py`, `dcrr.py`, `ctb.py`, `dcp.py`, `heads.py`, `hooks.py`, `types.py`, `interfaces.py`.

---

### `src/` (Core Research & Colab Engine)
- `model.py`: `MediTriageTransformer`, `FocalLoss`, and `JointLoss`.
- `trainer.py`: Research trainer with Colab T4 detection, AMP, and early stopping.
- `data_pipeline.py`: Comprehensive data compilation, stratified splitting, and seeding routines.
- `duplicate_validator.py`: Perceptual hashing and string distance deduplication.
- `schema.py`: `TriageSchema` enforcing `patient_presentation`, `department`, and `severity`.
- `checkpoint_manager.py`: Safe `.pt` loading with configuration compatibility checks.
- `metrics.py`: 22KB comprehensive metric suite for clinical classification.

---

### `scripts/` (Operational Entrypoints)
- `train.py`: Single-GPU training script with config hydration.
- `train_ddp.py`: Distributed multi-GPU (NCCL) trainer with Rank-0 shielding.
- `run_experiment.py`: Interactive/automated experiment runner across the 5 Model Zoo backbones.
- `colab_train.py`: Google Colab T4 self-contained training script.
- `evaluate.py`: Standalone checkpoint evaluator and prediction exporter.
- `serve_api.py`: FastAPI inference microservice with Basic Auth and safety rules.
- `serve_dashboard.py` / `export_dashboard_data.py`: Local web dashboard data pipeline and server.
- `reproduce_paper.py`: Automated reproduction script for publication metrics.
