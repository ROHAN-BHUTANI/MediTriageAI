# MediTriageAI — Authoritative Training Configuration Audit

**Specification Baseline:** `v1.0.0-FROZEN`  
**Audit Date:** `2026-08-16`  
**Inspected Source:** [src/config_manager.py](file:///c:/Users/bhuta/Desktop/MediTriageAI_Data_Engine/src/config_manager.py), [src/trainer.py](file:///c:/Users/bhuta/Desktop/MediTriageAI_Data_Engine/src/trainer.py)

---

## 1. Ground-Truth Hyperparameters & Runtime Configurations

| Parameter Name | Source Location | Ground-Truth Value | Description / Constraint |
|---|---|---|---|
| **Learning Rate (Heads)** | `src/config_manager.py:25` | `1e-4` | Learning rate for specialist & severity classification heads |
| **Encoder Learning Rate** | `src/config_manager.py:26` | `2e-5` | Differential learning rate for pretrained transformer encoder |
| **Optimizer** | `src/trainer.py:283` | `AdamW` | Decoupled weight decay optimizer |
| **Weight Decay** | `src/config_manager.py:28` | `0.01` | L2 regularization parameter |
| **LR Scheduler** | `src/trainer.py:307` | `CosineAnnealingWithWarmup` | Linear warmup followed by cosine decay |
| **Warmup Ratio** | `src/config_manager.py:30` | `0.1` (10% of total steps) | Warmup fraction |
| **Batch Size (Per-GPU)** | `src/config_manager.py:32` | `32` (DGX) / `16` (Local) | Micro-batch size per device |
| **Gradient Accumulation** | `src/config_manager.py:34` | `2` (Effective batch: 64) | Gradient accumulation steps |
| **Epochs** | `src/config_manager.py:36` | `10` | Total training epochs |
| **Max Sequence Length** | `src/config_manager.py:38` | `512` | Tokenizer max sequence length |
| **Mixed Precision (AMP)** | `src/trainer.py:338` | `torch.amp.autocast(fp16/bf16)` | Automatic Mixed Precision on CUDA |
| **Gradient Clipping** | `src/trainer.py:372` | `1.0` (max norm) | Prevents gradient explosion |
| **Early Stopping** | `src/trainer.py:410` | `Patience = 3 epochs` | Monitors validation Macro-F1 |
| **Checkpoint Frequency** | `src/trainer.py:425` | Every 1 epoch | Saves best checkpoint and latest state |
| **Global Random Seed** | `src/data_pipeline.py:45`| `42` | Seed for Python, NumPy, PyTorch CPU/CUDA |
| **DataLoader Workers** | `src/config_manager.py:48`| `4` (DGX) / `0` (Windows Local)| Worker subprocesses |
| **Persistent Workers** | `src/config_manager.py:50`| `True` (when workers > 0) | Keeps worker processes alive across epochs |
| **Pin Memory** | `src/config_manager.py:52`| `True` (when CUDA available) | Enables fast host-to-device memory copies |
| **Device Selection** | `src/trainer.py:207` | `cuda:LOCAL_RANK` or `cpu` | Auto-detects local rank for DDP or single GPU |
