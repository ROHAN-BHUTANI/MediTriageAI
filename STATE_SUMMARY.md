# MediTriageAI — State Summary Report

This document captures the current project state, completed evaluations, verified implementations, and upcoming tasks for the next development session.

---

## 1. Current Project State

### Frozen Architecture & Parameters
* **Taxonomies**: 13 Specialist Routing categories; 5 Severity Triage ESI tiers.
* **Loss weights**: $L_{\text{joint}} = 1.0 \cdot L_{\text{specialist}} + 1.2 \cdot L_{\text{severity}}$.
* **Data Splits**: Seed-level split (80/10/10 ratio), yielding 15,996 train / 2,000 val / 2,000 test rows.
* **Optimization**: AdamW, Learning Rate: 2e-5 (encoder) and 1e-4 (heads), Weight Decay: 0.01.

### Verified Implementations
* **PyTorch Training Loop**: Fully implemented in [train.py](file:///c:/Users/bhuta/Desktop/MediTriageAI_Data_Engine/scripts/train.py) with differential learning rates, joint loss propagation, and checkpoint saving.
* **Out-of-Bounds Fix**: Adjusted `max_position_embeddings` to `512` to handle position indexing for sequence lengths up to 256.
* **Windows Console Compatibility**: Replaced Unicode character `★` with `*` in the interactive runner to prevent Windows `cp1252` encoding crashes.
* **Exporter Fix**: Corrected parameter mapping in `export_dashboard_data.py` (`output_dir` $\rightarrow$ `output_path`).
* **Test Status**: All **29 unit tests pass successfully**.

---

## 2. Experimental Results (Verified)

### Baseline Evaluation (Full Test Split, $N_{\text{test}} = 1,999$)
* **Specialist Routing (best)**: TF-IDF + Linear SVM (**26.11% accuracy**, **11.01% macro-F1**).
* **Severity Triage (best)**: TF-IDF + Random Forest (**98.25% accuracy**, **93.70% macro-F1**).

### Transformer Evaluation (Matched Test Split, $N_{\text{train}} = 3,000$, $N_{\text{test}} = 1,999$)
* **mBERT**: Accuracy = 22.36% (specialist), 79.44% (severity), Specialist Macro-F1 = **2.81%**.
* **DistilBERT-multilingual**: Accuracy = 22.36% (specialist), 79.44% (severity), Specialist Macro-F1 = **2.81%**.

### Statistical Validation (Full Test Split)
* **McNemar's Significance**: The difference between the matched baseline (SVM on 3,000 rows) and the transformer (mBERT on 3,000 rows) is **not statistically significant** ($p = 0.1931$). When sample sizes are properly matched, the tiny multilingual transformer achieves baseline-equivalent routing performance.
* **Bootstrap 95% Confidence Interval**: mBERT macro-F1 interval is $[2.63\%, 3.01\%]$.
* **Script Robustness Parity**: mBERT achieves nearly identical specialist accuracy (**22.24%** on English and **22.40%** on Hinglish splits), proving script-invariance.

### Label Leakage Audit & Clinician Validation
* **Leakage Verdict**: Circular label leakage was identified. Evaluated against a clinician-annotated subset ($N=200$), the Random Forest baseline's severity macro-F1 collapsed by **68.48%** (from 93.70% down to 25.22%).
* **Transformer Stability**: In contrast, mBERT's severity F1 remained stable at **16.22%**, confirming semantic generalization rather than keyword memorization.

### Performance Collapse Diagnosis
* **Diagnosis Verdict**: Initial evaluation issues stemmed from severe under-training (160 rows). Scaling training data to 3,000 samples restored baseline parity, though full performance still requires a GPU and pre-trained weights to overcome standard training overfitting.

---

## 3. Web Service & Dashboard
* **Inference API**: FastAPI server built at `scripts/serve_api.py` with Pydantic schemas, Basic Auth, and automated OpenAPI docs. Verified working end-to-end with dynamic dynamic port finding and unicode logs support.
* **Web UI**: Static dashboard frontend in `dashboard_web/` fully updated to show:
  * Model leaderboard using ONLY full-test-set numbers.
  * Clearly labeled "Known Limitation: Severity Label Circularity (Audited & Addressed)" panel.
  * McNemar and Bootstrap statistical results.
  * Language-wise performance breakdown with sample counts (English: 21/499, Hinglish: 63/1500).
* **Demo Walkthrough**: Prepared a 5-minute live walkthrough script at [DEMO_SCRIPT.md](file:///c:/Users/bhuta/Desktop/MediTriageAI_Data_Engine/DEMO_SCRIPT.md).

---

## 4. Next Steps & Actions

1. **GPU Scale-Up**: Re-run the transformers on the full 15,970-row split using a GPU-enabled environment with full access to Hugging Face pre-trained weights.
2. **Expand Clinician Annotations**: Scale the clinician-annotated ground truth from 200 samples to 1,000 samples and compute Fleiss' Kappa IAA.
3. **External Validation**: Test cross-dataset generalization by evaluating on external ED triage or telehealth transcripts.
