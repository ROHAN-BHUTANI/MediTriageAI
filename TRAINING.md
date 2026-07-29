# MediTriageAI Training Guide

The central entry point for all training experiments is the deterministic `run_experiment.py` script. 

## Training Modes

The experiment runner enforces strict training modes to prevent accidental full-scale execution during development:

1. **Smoke**: Operates on a highly constrained batch sample limit. Bypasses metrics storage to validate end-to-end pipeline wiring.
2. **Development**: Uses a 10% stratified sample of the validation and training set. Validates optimization configurations over a smaller footprint.
3. **Publication**: Full-scale training spanning the 7.6M dataset. Includes strict checkpointing, full evaluation, and dashboard metric exports.

## Usage

```bash
# Execute the central orchestrator
python -m scripts.run_experiment --mode smoke
```

When prompted, input `5` to run the novel E-PATH-CO-REASON model.

## Orchestration

The orchestrator guarantees:
1. Deterministic seeded environments.
2. Cross-model evaluation using exactly the same split slices.
3. Automated saving of optimal weights to `results/<model_name>/checkpoint.pt`.