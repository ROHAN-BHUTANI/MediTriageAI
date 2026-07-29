# MediTriageAI — Final Frozen Architecture

This document defines the frozen specifications and architectural choices for the MediTriageAI research (Track A) and product demo (Track B).

---

## 1. Task Definitions & Label Taxonomies

- **Specialist Routing Head (13 classes)**:
  - Label Set: `CARDIO_PULM`, `ED`, `ENT_OPHTHALMO`, `GEN_MED`, `GI`, `NEURO`, `OBGYN`, `ONCOLOGY_HEME`, `ORTHO`, `PEDS`, `PSYCH`, `RENAL_URO`, `SURGERY`.
  - Objective: Route patient complaints to the most appropriate clinical specialty.
- **Severity Triage Head (5 classes)**:
  - Label Set: `S1` (Resuscitation), `S2` (Emergent), `S3` (Urgent), `S4` (Less Urgent), `S5` (Non-Urgent).
  - Objective: Assign an Emergency Severity Index (ESI) score based on symptom clinical urgency.

---

## 2. Dataset Pipeline & Grouped Split

- **Dataset Size**: 19,996 rows synthesized from 4,999 raw MTSamples seed records.
- **Hinglish Perturbation**: Grounded in the Bhargava et al. (2018) phonetic transliteration framework, generating deterministic, seed-reproducible Hinglish orthographic variants.
- **Grouped Split Ratio**: 80% train, 10% validation, 10% test.
- **Data Leakage Mitigation**: Grouped split strictly partitioned at the seed-document level (`seed_id`) to ensure no variants of a single patient record span across train/val/test splits.
- **Sequence Length**: Max token length is frozen at `256` to balance GPU memory limits with the retention of detailed patient history.

---

## 3. Neural Architecture & Loss Function

- **Encoder Backbone**: Hard parameter sharing using `XLM-RoBERTa-large` as the primary encoder, with parallel evaluations against three baseline encoders:
  1. `mBERT` (`bert-base-multilingual-cased`)
  2. `DistilBERT-multilingual` (`distilbert-base-multilingual-cased`)
  3. `IndicBERT` (`ai4bharat/indic-bert`)
- **Classification Heads**: Two parallel linear classification layers connected to the pooled `[CLS]` token representation.
- **Joint Loss Formulation**:
  $$L_{\text{total}} = \alpha \cdot L_{\text{specialist}} + \beta \cdot L_{\text{severity}}$$
  - Specialist weight ($\alpha$): `1.0`
  - Severity weight ($\beta$): `1.2`
  - *Justification*: Severity has direct clinical safety implications, thus carries a slightly higher weight to penalize severity sorting errors.

---

## 4. Model Training & Optimization Hyperparameters

- **Optimizer**: AdamW with weight decay of `0.01` and epsilon of `1e-8`.
- **Learning Rates**: Encoder fine-tuning rate of `2e-5`; classification heads training rate of `1e-4`.
- **Batch Size**: `8` per GPU (with gradient accumulation of 2 if training on 16GB VRAM hardware).
- **Epochs**: `5` epochs with early stopping on validation loss (patience = 1).
- **Learning Rate Scheduler**: Linear decay scheduler with a warmup phase of 10% of total training steps.

---

## 5. Confidence Calibration & Verification

- **Calibration Method**: Post-hoc Temperature Scaling applied to the classification logits to align confidence values with empirical accuracy.
- **Evaluation Metrics**:
  - Macro-F1 score (both heads)
  - Expected Calibration Error (ECE)
  - Severity Adjacent Confusion Rate (adjacent ESI levels)
  - Cohen's Weighted Kappa (ordinal evaluation of severity)
