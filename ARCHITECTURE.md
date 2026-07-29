# MediTriageAI Architecture

MediTriageAI is a robust, production-ready research pipeline designed for end-to-end dataset ingestion, multi-task model training, and clinical evaluation.

## Core Components

The architecture is composed of three tightly coupled modules:

### 1. Data Engine (`meditriage/builder`)
The core data engine is a deterministic, 7-stage streaming builder that standardizes heterogeneous data sources into a unified clinical schema.
- **Stage 1: Ingest** - Load data from adapters.
- **Stage 2: Schema Align** - Map raw columns to the canonical format `["id", "split", "dataset_source", "language", "raw_text", "department", "triage_level"]`.
- **Stage 3: Normalize** - Apply text normalization and encoding.
- **Stage 4: Deduplicate** - Globally deduplicate records via MD5 hashing.
- **Stage 5: Filter** - Remove low-quality or invalid samples.
- **Stage 6: Partitions** - Compute and stratify train/val/test splits.
- **Stage 7: Export** - Export the unified dataset to Parquet format.

### 2. Model Zoo (`models/`)
The registry provides isolated plug-and-play models adhering to the `BaseCheckpointRegistry` interface.
- **`emergent_path_triage` (E-PATH-CO-REASON)**: A multi-task transformer (e.g. XLM-RoBERTa-large) customized for joint prediction of clinical specialist routing (Department) and clinical severity (Triage Level).

### 3. Training & Evaluation Pipeline (`scripts/`)
- `run_experiment.py`: Centralized orchestrator supporting deterministic runs.
- `train.py`: The PyTorch training loop integrating dynamic learning rates, joint loss propagation, and checkpointing.
- `evaluate.py`: Computes macro-F1, adjusted error rates, and generates metrics.

## Design Philosophy
- **Portability**: Completely location-independent execution paths.
- **Determinism**: Fully seeded PRNGs ensuring repeatable results (see `REPRODUCIBILITY.md`).
- **Strict Schema Enforcement**: Schema consistency enforced at ingestion rather than load-time.
