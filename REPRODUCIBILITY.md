# Reproducibility Framework

Scientific rigor is a core tenet of this framework. MediTriageAI ensures reproducibility through several strict constraints:

## 1. Deterministic Checksums
Every time a dataset is compiled, the `builder` emits a hash of the combined data manifest. During training, the `EmergentTrainer` writes this exact dataset fingerprint to `dataset_manifest.json`. If you resume a checkpoint using a modified dataset, the trainer will explicitly crash and halt.

## 2. Seed Injection (DDP Safe)
In `train_ddp.py`, random seed states are shielded per-rank. The global seed is parsed from the YAML configuration, and local seeds are explicitly set to `global_seed + rank` to avoid DDP broadcast sync drift.

## 3. Configuration Subsystem
Hardcoding variables inside execution scripts is strictly prohibited. `src/config_manager.py` defines the `TrainingConfig` dataclass. The config object is exported exactly as it was hydrated into JSON alongside model checkpoints, allowing an observer to recreate the precise configuration state matching a given `.pt` file.
