# V7 Transformer Attempt - Aborted (Timeout)

## Objective
Retrain `DistilBERT-multilingual` on specialist routing (severity axis excluded) with the following specifications:
- `max_length` restored to `128` (up from 64).
- Linear warmup + differential learning rates (Encoder: `2e-5`, Classifier: `1e-4`).
- Training to actual convergence (early stopping based on validation loss).
- Run on both the original 13-class labels and the proposed 5-class consolidated taxonomy.

## Execution Log & Status
- **Attempt 1:** The script was initiated with `batch_size=32`. The Intel GPU immediately threw an Out-Of-Memory (OOM) exception due to the doubled `max_length=128`.
- **Attempt 2:** The batch size was reduced to `8`. The Intel GPU still threw an OOM exception (`[W714 11:11:31.000000000 dml_heap_allocator.cc:120] DML allocator out of memory!`).
- **Attempt 3:** The batch size was reduced to `2`, and gradient accumulation steps were introduced (`8` steps, simulating a batch size of 16). 
  - The script successfully initialized the GPU and began the first training step.
  - *LR Verification (Step 0):* `[orig_13] Step 0 LRs: Encoder=0.00e+00, Classifier=0.00e+00` (Warmup correctly initialized at 0).
  - *Time limit breached:* The script ran for over 20 minutes but failed to complete even a single epoch (out of up to 15 epochs) for the first training loop (13-class original). 

## Conclusion
The V7 training pass was **aborted** because it would definitively exceed the 90-minute hard time budget. 

**Root Causes for Abort:**
1. **Memory Constraints:** Restoring `max_length=128` forced the batch size down to `2` to prevent Intel GPU DirectML OOM errors.
2. **Computational Overhead:** The combination of `batch_size=2` with gradient accumulation over 16,000 training rows translates to 8,000 forward passes and 1,000 backward passes per epoch. On this hardware, a single epoch takes >25 minutes. Two full training runs (13-class and 5-class) with early stopping would take upwards of 10+ hours.

Without dedicated heavy-duty GPU hardware (e.g., A100/V100) or aggressive quantization, a `max_length=128` Transformer pass over this dataset cannot be achieved within a 90-minute time constraint.
