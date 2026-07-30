# Training Guide

This guide walks you through executing training routines on single-GPU nodes or multi-GPU DGX clusters.

## 1. Preparation
Before running the trainer, ensure the dataset is generated and normalized.
```bash
python scripts/run_baseline.py --export-format parquet
```

## 2. Configuration
All hyperparameters are read from a YAML config file. Create a file `configs/experiment_dgx.yaml` (see `EXPERIMENT_TEMPLATE.yaml` for a reference) and set paths for the `dataset_manifest.json` and dataset files.

## 3. Single-GPU Training
If you are testing or doing small ablations, use the standard `train.py` wrapper:
```bash
python scripts/train.py --config configs/experiment_dgx.yaml
```

## 4. Multi-GPU Distributed Training (DGX)
For production runs on DGX clusters, use `torchrun` and `scripts/train_ddp.py`:
```bash
torchrun --nproc_per_node=4 scripts/train_ddp.py --config configs/experiment_dgx.yaml
```
- **Rank 0 Shielding**: Only Rank 0 will save checkpoints and export metrics, ensuring safe file I/O.
- **Tuning Performance**: Set `use_torch_compile: true` in your YAML for major throughput gains. Ensure `gradient_checkpointing` is `true` if you encounter CUDA OOM errors.

## 5. Artifacts
Outputs are exported to the directory specified in your YAML (`checkpoint_dir`). This includes:
- `checkpoint_epoch_X.pt`: Weights and optimizer state
- `dataset_manifest.json`: Fingerprint of dataset for resumption
- `predictions.parquet`: Multi-task predictions on validation set
- `training_metadata.json`: Full log of parameters, seeds, and hardware profile.
