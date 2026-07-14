# Slicing Bug Resolution & New Subset Distributions

The data-slicing bugs (using `.head()` and `.iloc[:]` on an unshuffled department-sorted dataset) have been completely removed from the codebase. 

## 1. Implementation
A shared `create_stratified_subset(df, n, label_col)` function was added to `src/sampling.py`. It uses scikit-learn's `train_test_split(stratify=y)` to guarantee subsets perfectly match the full dataset's class distribution.

All ad-hoc slicing calls were replaced:
- `src/dataset.py`: Updated `load_split_rows` to optionally take `max_rows` and use the stratify function. 
- `scripts/train.py`: Stripped out inline array slicing; now passes `max_rows` into the robust dataset loader.
- `scratch/statistical_validation_v2.py`: Replaced `.iloc[:3000]` with the stratify function.
- `scratch/compile_clinician_subset.py`: Replaced `.iloc[:200]` with the stratify function.

Unit tests were added in `tests/test_sampling.py` which verify that a 1000-sample subset guarantees representation across all 13 classes and caps subset proportions accurately. **The tests pass.**

## 2. New 3,000-Sample Subset Distribution

This is the class distribution the transformers and baselines will **actually** see in the upcoming retraining step. Notice that every class is now correctly represented.

### Specialist (department_code)
| Department Code | Sample Count (3,000 Subset) |
| :--- | :--- |
| **GEN_MED** | 937 |
| **SURGERY** | 681 |
| **ORTHO** | 255 |
| **CARDIO_PULM** | 240 |
| **NEURO** | 189 |
| **GI** | 153 |
| **RENAL_URO** | 148 |
| **ENT_OPHTHALMO** | 129 |
| **OBGYN** | 88 |
| **ONCOLOGY_HEME** | 58 |
| **ED** | 47 |
| **PEDS** | 41 |
| **PSYCH** | 34 |

### Severity (severity_heuristic)
| Severity Level | Sample Count (3,000 Subset) |
| :--- | :--- |
| **S4** | 2,420 |
| **S5** | 411 |
| **S2** | 75 |
| **S3** | 54 |
| **S1** | 40 |

> [!NOTE]
> All 13 specialist classes and all 5 severity levels are now present in the training subset. We are ready to proceed with a clean, unbiased retraining phase.
