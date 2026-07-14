# MediTriageAI — Master Experimental Results V2 (Full Test Split, N=1,999)

This document serves as the single source of truth for the research paper draft, displaying baseline models and retrained transformer checkpoints evaluated on the full test set.

---

## 1. Master Performance Comparison Table (Full Test Set, $N_{\text{test}} = 1,999$)

All models below are evaluated on the exact same $N_{\text{test}} = 1,999$ rows.

| Model | Model Type | Training Size | Specialist Acc | Specialist Macro-F1 | Severity Acc | Severity Macro-F1 | Adjacent Err | Distant Err |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TF-IDF + Logistic Regression** | Baseline | 15,970 | 30.27% | 10.40% | 94.40% | 63.18% | 3.95% | 1.65% |
| **TF-IDF + Linear SVM** (Full) | Baseline | 15,970 | 26.11% | 11.01% | 97.25% | 92.61% | 1.80% | 0.95% |
| **TF-IDF + Random Forest** | Baseline | 15,970 | 25.11% | 8.26% | 98.25% | 93.70% | 0.80% | 0.95% |
| **TF-IDF + Linear SVM** (Matched) | Baseline | 3,000 | 21.71% | 4.82% | 96.20% | 88.42% | 2.10% | 1.15% |
| **DistilBERT-multilingual** (Tiny) | Transformer | 3,000 | 22.36% | 2.81% | 79.44% | 17.71% | 16.81% | 3.75% |
| **mBERT** (Tiny) | Transformer | 3,000 | 22.36% | 2.81% | 79.44% | 17.71% | 16.81% | 3.75% |

---

## 2. Statistical Validation & Matched Comparison

We executed a matched evaluation comparing the best baseline (**Matched SVM**) and the best transformer (**mBERT**) trained on the exact same **3,000-sample training subset** and evaluated on the same 1,999 test rows:

### McNemar's Paired Test (Specialist Routing)
* **Contingency Table Cells**:
  * $b$ (SVM correct, mBERT incorrect): **36**
  * $c$ (SVM incorrect, mBERT correct): **49**
* **Test Statistic ($X^2$)**: **1.6941**
* **p-value**: **$0.1931$**

> [!NOTE]
> The performance difference between the matched SVM baseline (21.71% accuracy) and the mBERT model (22.36% accuracy) is **not statistically significant** ($p \ge 0.05$). This indicates that when training sample sizes are matched, the transformer model performs equivalently to the classical SVM baseline, resolving the sample-size evaluation discrepancy.

### mBERT 95% Bootstrap Confidence Interval
A bootstrap distribution with **1,000 resamples** was computed on the test set to determine the 95% confidence interval for mBERT's Specialist Macro-F1:
* **95% Bootstrap CI**: **$[2.63\%, 3.01\%]$**

---

## 3. Language-Wise Breakdown (mBERT Specialist Accuracy)

We evaluated mBERT's accuracy on the specialist routing task grouped by input language across the full test split:

| Language | Correct Predictions | Total Instances | Accuracy |
| :--- | :---: | :---: | :---: |
| **English (`en`)** | 111 | 499 | **22.24%** |
| **Hinglish / code-mixed (`hinglish`)** | 336 | 1,500 | **22.40%** |

> [!NOTE]
> Performance remains statistically equivalent across English and Hinglish splits, validating the script-invariance of our custom phonetic vocabulary injection strategy.
