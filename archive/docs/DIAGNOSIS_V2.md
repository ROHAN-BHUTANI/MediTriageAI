# Diagnosis of the Below-Random-Chance Result (2.81% Macro-F1)

## 1. Predicted Label Distribution (mBERT Specialist Head)
The predicted label distribution for mBERT on the full test set is completely degenerate. The classification report shows that the model predicted class `12` (`SURGERY`) for **100% of the 1,999 test samples**. It did not predict any other class even once. 

## 2. Label Encoding Audit
There is **no silent mismatch** in the label encoding mapping. Both `src/dataset.py` (which creates the PyTorch dataset) and `scratch/statistical_validation_v2.py` (which runs the matched-size evaluation) correctly map string labels to indices and back using the exact same `SPECIALIST_CLASSES` list in `src/model.py`. 

## 3. Epochs and Convergence
The 3,000-sample retraining ran for exactly 2 epochs. The logs show:
- **mBERT Train Loss**: Decreased (1.88 -> 0.99)
- **mBERT Val Loss**: Increased (4.57 -> 5.31)

This indicates rapid **overfitting**, not under-convergence. The model quickly memorized the training data and began failing to generalize to the validation set. 

## 4. True Class Distribution (The Root Cause)
The full test set is reasonably distributed across all 13 classes (e.g., GEN_MED: 628, SURGERY: 448, ORTHO: 180, etc.). 

However, inspecting the **3,000-sample matched training subset** reveals a catastrophic class imbalance caused by the data slicing method (`df.iloc[:3000]` and `df.head(3000)` without shuffling). The dataset appears to be sorted/clustered by department. The first 3,000 training rows contain exactly:
- `SURGERY`: 2,418 (80.6%)
- `RENAL_URO`: 526 (17.5%)
- `GI`: 24 (0.8%)
- `CARDIO_PULM`: 24 (0.8%)
- `GEN_MED`: 8 (0.3%)
- **0 examples** for the remaining 8 classes (ORTHO, ED, PEDS, etc.)

## 5. Verdict
This is a **data slicing bug causing catastrophic class imbalance**, heavily amplified by the mechanics of the macro-F1 metric. 

Because both the SVM and Transformers were trained on a subset that was 80.6% `SURGERY`, they learned to exclusively predict `SURGERY`. On the test set, they got a decent F1 score on the `SURGERY` class (~36.5%), but exactly **0%** on the other 12 classes. Since macro-F1 averages the score equally across all 13 classes regardless of their frequency, the final score becomes `(36.5% + 0 + 0 + ... ) / 13 = 2.81%`. 

The matched size (3,000) was a good idea, but it must be an evenly stratified or fully randomized sample, not the first 3,000 rows of an unsorted dataframe. This invalidates the current 2.81% and 4.82% metrics.
