# Full Codebase Data-Slicing Audit

A systematic scan of the repository for dataframe/list slicing (e.g., `.head()`, `.iloc[:]`, `[:max_rows]`) uncovered a pervasive data-slicing bug. Because the underlying dataset is naturally clustered/sorted by department, taking the top $N$ rows without shuffling creates pathologically imbalanced subsets.

Here is the itemized report of every compromised slicing event.

## 1. The Original Phase 3 Training (160-sample split)
- **File**: `scripts/train.py`
- **Line**: 68 (`rows = rows[:max_rows]`)
- **Shuffled before slicing?**: **No**. The dataset loader (`load_split_rows` in `dataset.py`) loads the split sequentially, and the `train.py` script slices the first $N$ rows before shuffling for the dataloader.
- **Resulting Distribution (Train, 160 rows)**:
  - **Specialist**: `RENAL_URO` (104), `GI` (24), `CARDIO_PULM` (24), `GEN_MED` (8). **(9 classes missing entirely)**
  - **Severity**: `S4` (148), `S5` (8), `S3` (4). 
- **Resulting Distribution (Test, 160 rows)**:
  - **Specialist**: `SURGERY` (88), `RENAL_URO` (60), `GI` (8), `ENT_OPHTHALMO` (4). 
  - **Severity**: `S4` (152), `S5` (4), `S3` (4).
- **Impact**: **Compromised**. The original Phase 3 transformer models were trained predominantly on `RENAL_URO`, but evaluated predominantly on `SURGERY` (which they had never seen). The models collapsed to predicting `RENAL_URO` blindly, artificially scoring well on the test subset only because `RENAL_URO` happened to overlap. 

## 2. Phase 7 Retraining (3,000-sample matched split)
- **File**: `scratch/retrain_zoo.py`
- **Line**: 40 (`max_rows=3000` passed to config, executing `rows[:max_rows]`)
- **Shuffled before slicing?**: **No**. 
- **Resulting Distribution (Train, 3000 rows)**:
  - **Specialist**: `SURGERY` (2418), `RENAL_URO` (526), `GI` (24), `CARDIO_PULM` (24), `GEN_MED` (8). **(8 classes missing entirely)**
- **Impact**: **Compromised**. Caused the 2.81% below-random-chance macro-F1 collapse, as the model learned to blindly predict `SURGERY` (80.6% of its training data).

## 3. Phase 7 SVM Matched Baseline
- **File**: `scratch/statistical_validation_v2.py`
- **Line**: 64 (`train_df_sub = train_df.iloc[:3000].copy()`)
- **Shuffled before slicing?**: **No**. 
- **Resulting Distribution**: Exactly identically flawed as the 3,000-sample transformer split (80.6% `SURGERY`). 
- **Impact**: **Compromised**. Brought the classical SVM's macro-F1 plummeting to 4.82% by forcing it to train on the same broken subset.

## 4. Phase 7 Clinician Ground Truth Subset
- **File**: `scratch/compile_clinician_subset.py`
- **Line**: 70 (`clinician_df = test_df.iloc[:200].copy()`)
- **Shuffled before slicing?**: **No**. 
- **Resulting Distribution (Test, 200 rows)**:
  - **Specialist**: `SURGERY` (128), `RENAL_URO` (60), `GI` (8), `ENT_OPHTHALMO` (4). 
- **Impact**: **Compromised**. The manual clinician evaluation (which uncovered the label leakage) evaluated the models on a heavily biased subset (64% `SURGERY`). While the leakage finding itself (Random Forest exploiting keywords) remains directionally true, the absolute F1 metrics reported for that clinician subset are strictly bound to this imbalanced slice.

---

### Conclusion & Verdict
**Every single subset evaluation in the project's history—from the very first 160-sample training run in Phase 3 up to the Phase 7 fixes—has been poisoned by this identical unshuffled-slicing bug.**

All absolute numbers and F1 scores tied to a "subset" (`160`, `3000`, or `200` clinician samples) are invalid because none of these subsets were drawn from the true data distribution. 
