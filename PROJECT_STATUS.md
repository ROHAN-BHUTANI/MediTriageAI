# Project Status

## Repository State: FROZEN
The MediTriageAI Data Engine and Modeling Repository has completed its Phase 5 roadmap and is now in **Repository Freeze Mode**. No new features will be introduced to the core `1.0.x` baseline.

## Implemented Features
- **Datasets**: Unified 13+ distinct text-based medical triage datasets into a common format (patient presentation -> department & severity).
- **Architecture**: `MediTriageTransformer`, a dual-head transformer (currently using `xlm-roberta-base`) designed for multi-task predictions.
- **Training Pipeline**: Comprehensive `EmergentTrainer` capable of gracefully masking unannotated data, thereby allowing disjoint datasets to collaboratively inform the model.
- **Evaluation Pipeline**: Detailed metric calculation including Macro F1, Weighted F1, AUROC, and class-wise prediction exports (`misclassified.csv`, `entropy_distribution.csv`).
- **Distributed Training**: Fully operational DDP training (`train_ddp.py`) ready for scaling across multi-GPU environments with AMP and gradient checkpointing.
- **Explainability**: Integrated baseline heuristic tracking and entropy measurements over predictions.

## Remaining Limitations
- **Sequence Truncation**: Extremely long clinical notes are currently truncated to the model's maximum sequence length (512 tokens), potentially discarding trailing contextual information.
- **Label Granularity**: Some datasets provided highly-specific specialty information that had to be generalized (e.g., mapped to broad buckets like `Surgery`) resulting in some loss of granularity.
- **FSDP**: While DDP is implemented and highly performant for a single 8-GPU node, multi-node scaling across hundreds of GPUs via DeepSpeed or FullyShardedDataParallel (FSDP) has not yet been integrated.

## Future Work
- Transitioning the backbone from an encoder-only model to an instruction-tuned decoder (e.g. Llama-3, Mistral) for generative triage rationalization.
- Integrating external knowledge graph embeddings (e.g., SNOMED-CT) directly into the tokenization phase.
- Multi-modal support (e.g., ingesting triage ECG traces alongside text).
