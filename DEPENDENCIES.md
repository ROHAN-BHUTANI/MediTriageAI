# Dependencies

The MediTriageAI Data Engine requires the following core dependencies:

## Core Libraries
- **Python**: 3.10+
- **PyTorch**: >= 2.0.0 (CUDA 11.8 or 12.1 recommended for DGX/A100 runs)
- **Transformers**: >= 4.30.0 (HuggingFace)
- **Datasets**: >= 2.14.0 (HuggingFace)

## Data Processing
- **Pandas**: >= 2.0.0
- **NumPy**: >= 1.24.0
- **Scikit-learn**: >= 1.3.0
- **Polars**: >= 0.19.0 (For large-scale dataset normalization)

## Distributed Training
- **Accelerate**: >= 0.22.0
- **DeepSpeed**: >= 0.10.0 (Optional, for FSDP)
- **NCCL**: Included with PyTorch for DistributedDataParallel (DDP).

## Utilities
- **Pytest**: >= 7.4.0 (For testing)
- **Black**: >= 23.0.0 (For formatting)
- **Ruff**: >= 0.1.0 (For linting)

To install all dependencies, run:
```bash
pip install -r requirements.txt
```
