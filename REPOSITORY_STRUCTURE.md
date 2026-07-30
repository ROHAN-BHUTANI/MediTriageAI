# Repository Structure

```text
MediTriageAI_Data_Engine/
├── src/                        # Core Library
│   ├── config_manager.py       # YAML configuration parsing & dataclass definition
│   ├── data_ingestion.py       # Data fetching abstractions
│   ├── dataset_adapters.py     # Source-specific dataset standardizers
│   ├── duplicate_validator.py  # Perceptual hashing deduplication
│   ├── schema.py               # TriageSchema standard enforcement
│   ├── data_pipeline.py        # Central data compilation & export routine
│   ├── model.py                # Dual-head transformer model
│   ├── trainer.py              # Custom multi-task trainer loop
│   ├── evaluation.py           # Metric calculation (Macro F1, AUROC)
│   └── dataset_registry.py     # Registry of supported dataset adapters
├── scripts/                    # Entrypoints & Executables
│   ├── run_baseline.py         # Main script to run the dataset builder
│   ├── train.py                # Single-GPU training script
│   ├── train_ddp.py            # Multi-GPU NCCL distributed training script
│   ├── evaluate.py             # Script to evaluate existing checkpoints
│   ├── serve_api.py            # FastAPI inference endpoint
│   └── launch_experiments.py   # Hyper-parameter sweep launcher
├── models/                     # Model-specific code and types
├── tests/                      # Unit and integration test suite
├── configs/                    # YAML configuration files
├── README.md                   # Main landing page
└── VERSION                     # Current stable release version
```
