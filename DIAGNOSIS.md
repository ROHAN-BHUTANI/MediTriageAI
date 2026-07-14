# MediTriageAI — Model Performance Collapse Diagnosis Report

This diagnosis report investigates the severe performance collapse of the multilingual transformer models (`XLM-RoBERTa-large` and `mBERT`) on the full 1,999-row test set.

---

## 1. Actual Training Row Counts in Phase 3

We audited the training configuration in [train.py](file:///c:/Users/bhuta/Desktop/MediTriageAI_Data_Engine/scripts/train.py) and [run_experiment.py](file:///c:/Users/bhuta/Desktop/MediTriageAI_Data_Engine/scripts/run_experiment.py):
* **Config Default**: `TrainingConfig` has a default parameter `max_rows: int | None = 160`.
* **Loader Function**: `_build_split_loader` truncates the loaded rows to `max_rows` if set:
  ```python
  if max_rows is not None and max_rows > 0:
      rows = rows[:max_rows]
  ```
* **Execution**: In `run_experiment.py`, `TrainingConfig` is instantiated with defaults (`trainer.TrainingConfig(model_cls=spec.model_cls)`), which defaults `max_rows` to `160`.
* **Verdict**: The models were indeed trained on exactly **160 rows** (instead of the full 15,996 training split). This is verified by checking the loader configurations.

---

## 2. Predicted Label Distribution on the Full Test Set ($N_{\text{test}} = 1,999$)

Evaluating the checkpoints on the full test set shows a **complete majority-class/single-class degenerate collapse** for both heads:

### A. Specialist Routing Head
* **XLM-RoBERTa-large**: Predicted **Class 7** for 1,999 out of 1,999 inputs ($100\%$ collapse).
  * *Result*: Specialist Accuracy = **0.15%** (exactly equal to the test support of Class 7: $3 / 1,999$).
* **mBERT**: Predicted **Class 11** for 1,999 out of 1,999 inputs ($100\%$ collapse).
  * *Result*: Specialist Accuracy = **4.20%** (exactly equal to the test support of Class 11: $84 / 1,999$).

### B. Severity Triage Head
* **Both Models**: Predicted **ESI Level S4** for 1,999 out of 1,999 inputs ($100\%$ collapse).
  * *Result*: Severity Accuracy = **79.44%** (exactly equal to the test support of `S4`: $1,588 / 1,999$).

This confirms that the models failed to learn clinical routing features and simply defaulted to predicting a single majority class for all inputs, explaining the near-zero macro-F1 scores.

---

## 3. Test Set Row Count Reconciliation (2,000 vs. 1,999)

The discrepancy between the raw split counts ($2,000$ test rows) and the actual loader counts ($1,999$ test rows) is explained by data cleansing in `load_split_rows` ([dataset.py](file:///c:/Users/bhuta/Desktop/MediTriageAI_Data_Engine/src/dataset.py)):
* **Cleansing Logic**: Rows with null/NaN values in the `text` column are dropped to prevent tokenizer execution crashes:
  ```python
  if df_split["text"].isna().sum() > 0:
      df_split = df_split.dropna(subset=["text"])
  ```
* **Dataset Null Analysis**:
  * **Test split**: Has exactly **1 null row** (reducing active test rows from $2,000$ to $1,999$).
  * **Train split**: Has exactly **26 null rows** (reducing active training rows from $15,996$ to $15,970$).
  * **Val split**: Has exactly **6 null rows** (reducing active validation rows from $2,000$ to $1,994$).

---

## 4. Diagnosis & Remediation Plan

### Primary Causes of Performance Collapse
1. **Severe Under-Training (Primary Cause)**: Training a complex multilingual transformer (e.g. mBERT with 110M parameters or XLM-R with 550M parameters) on only **160 samples** for only **2 epochs** is mathematically guaranteed to result under-fit. The model's loss landscape could not develop decision boundaries, causing it to collapse into predicting the majority class.
2. **Resource-Constrained CPU Limitations**: The previous session scaled down the dataset size to 160 because training full-sized transformer architectures on local CPU hardware is extremely slow. 

### Recommended Action Plan
To resolve this without running out of compute, we recommend:
1. **Switching to a lightweight ablation backbone**: Replace XLM-RoBERTa-large with **DistilBERT-multilingual** (which is significantly faster to train and evaluate on CPU).
2. **Evaluating on a resource-feasible training split**: Scale up training size to a mid-scale subset (e.g., $3,000$ to $5,000$ samples) rather than $160$.
3. **Using gradient accumulation & FP16**: Reduce peak memory load and simulate larger effective batches to ensure training stability.
