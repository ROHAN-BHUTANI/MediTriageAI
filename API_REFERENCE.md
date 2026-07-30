# API Reference

This document outlines the core public modules of the MediTriageAI Data Engine.

## `src.model.MediTriageTransformer`
The core neural architecture.
- **Parameters**: `encoder (nn.Module)` - A HuggingFace Transformer backbone.
- **Forward Pass**: Returns a tuple of `(specialist_logits, severity_logits)`.
  - `specialist_logits`: Shape `(batch_size, 13)`
  - `severity_logits`: Shape `(batch_size, 5)`

## `src.trainer.EmergentTrainer`
The training orchestrator.
- **Constructor Arguments**:
  - `model`: An instance of `MediTriageTransformer`.
  - `config`: A populated `TrainingConfig` dataclass.
  - `train_loader`, `val_loader`, `test_loader`: PyTorch DataLoaders returning input_ids, masks, department labels, severity labels.
  - `tokenizer`: HF Tokenizer for saving artifacts.
- **Methods**:
  - `train()`: Initiates the training loop.
  - `validate()`: Runs validation and metric collection (Macro-F1, AUROC).
  
## `src.dataset_adapters.DatasetAdapter`
The abstract base class for ingesting new datasets.
- **Abstract Methods**:
  - `clean()`: Must be implemented by subclasses. Maps source dataframe columns to `patient_presentation`, `department`, and `severity`.

## `src.config_manager.TrainingConfig`
A strongly typed dataclass defining the experiment.
- Contains hyper-parameters (LR, batch size), scaling arguments (`gradient_checkpointing`, `use_torch_compile`), and metadata (paths, seeds). Parseable from YAML via `from_yaml()`.
