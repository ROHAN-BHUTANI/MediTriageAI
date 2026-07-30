# System Architecture

The MediTriageAI Data Engine and Modeling framework is split into two tightly-coupled subsystems: the **Data Engineering Pipeline** and the **Multi-Task Trainer**.

## 1. Data Engineering Pipeline

The pipeline normalizes heterogeneous medical datasets into a unified schema for training.

- **Data Ingestion (`src/data_ingestion.py`)**: Fetches data from HF Datasets, local files, or Kaggle.
- **Dataset Adapters (`src/dataset_adapters.py`)**: Subclasses of a base adapter which apply dataset-specific cleaning routines. For instance, coercing triage variables into strict integers `(1-5)` and normalizing text formats.
- **Normalizer & Validator (`src/schema.py`, `src/duplicate_validator.py`)**: Applies the `TriageSchema` enforcing `patient_presentation`, `department` (13 specialties), and `severity` (1-5). Deduplicates using perceptual hashing across data sources to avoid cross-contamination.
- **Data Exporter (`src/data_pipeline.py`)**: Converts the final unified stream into CSV, JSON, or Parquet for ingestion by the training subsystem.

## 2. Multi-Task Trainer

The modeling subsystem is built upon HuggingFace Transformers and PyTorch DDP.

- **Config Manager (`src/config_manager.py`)**: Consolidates all hyper-parameters (LR, epochs, grad-acc, mask weights) into the `TrainingConfig` dataclass. Ensures no hard-coded variables exist in execution scripts.
- **Model (`src/model.py`)**: `MediTriageTransformer` encapsulates a base encoder (e.g. `xlm-roberta-base`) and branches into two independent classification heads:
  - Specialty Head (13 classes)
  - Severity Head (5 classes)
- **Trainer (`src/trainer.py`)**: The `EmergentTrainer` calculates masked multi-task losses. The loss functions ignore components where labels are `NaN` or un-annotated, allowing for joint learning across partially-labelled sources.
- **DDP Scaling (`scripts/train_ddp.py`)**: Implements `torch.distributed`, gradient checkpointing, `torch.compile`, and AMP for large-scale GPU scaling on DGX nodes. Rank-0 orchestration ensures safe I/O (checkpoints, metrics).
- **Evaluation (`src/evaluation.py`, `src/metrics.py`)**: Evaluates precision, recall, and Macro-F1 per class, returning `predictions.parquet` for analysis.
