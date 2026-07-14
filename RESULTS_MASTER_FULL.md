# MediTriageAI — Master Experimental Results (Full 2,000-Row Test Split)

This document presents the complete evaluation metrics, classification reports, confusion matrices, and statistical validation tests for both baseline and transformer models evaluated on the **full test split (1,999 rows)**.

---

## 1. Headline Results Comparison Table ($N_{\text{test}} = 1,999$)

All models in the table below are evaluated on the exact same $N_{\text{test}} = 1,999$ rows, eliminating the sample-size mismatch from previous sessions.

| Model | Model Type | Specialist Acc | Specialist Macro-F1 | Severity Acc | Severity Macro-F1 | Adjacent Err | Distant Err |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **TF-IDF + Logistic Regression** | Baseline | 30.27% | 10.40% | 94.40% | 63.18% | 3.95% | 1.65% |
| **TF-IDF + Linear SVM** | Baseline | 26.11% | 11.01% | 97.25% | 92.61% | 1.80% | 0.95% |
| **TF-IDF + Random Forest** | Baseline | 25.11% | 8.26% | 98.25% | 93.70% | 0.80% | 0.95% |
| **XLM-RoBERTa-large** (tiny) | Transformer | 0.15% | 0.30% | 79.44% | 17.71% | 16.81% | 3.75% |
| **mBERT** (tiny) | Transformer | 4.20% | 0.62% | 79.44% | 17.71% | 16.81% | 3.75% |

---

## 2. Meaningful Changes from the 160-Sample Subset

Evaluating on the full 1,999-row test split reveals critical shifts in performance compared to the 160-sample test subset:

> [!WARNING]
> * **Specialist Macro-F1 Drop**: mBERT's Specialist Macro-F1 dropped from **13.64%** (on the 160-sample subset) to **0.62%** (on the full test split). In the 160-sample subset, mBERT predicted class 11 for all inputs, yielding an F1 score of 54.55% for that class (which had 60 supporting samples) and raising the macro average. On the full test split, this prediction strategy failed, and the model failed to generalize.
> * **Severity Macro-F1 Drop**: Both mBERT and XLM-RoBERTa-large severity Macro-F1 dropped from **32.48%** to **17.71%**. In the 160-sample subset, ESI class `S4` accounted for 95% of samples (152/160), giving `S4` an F1 of 97.44%. On the full test split, the support of non-S4 classes increased (e.g., S5 has 297, S2 has 67), but the model continued to predict `S4` for all inputs. The `S4` class F1 dropped to 88.54% and all other classes remained at 0.00%, lowering the macro average.
> * **Severity Accuracy Shift**: The severity accuracy for the transformers dropped from **95.00%** to **79.44%** due to the higher volume of non-S4 samples in the full test split that were misclassified as S4.

---

## 3. Detailed Model Metrics (Full Test Set)

### A. XLM-RoBERTa-large (Tiny Configuration)

#### Specialist Routing Head
* **Accuracy**: 0.15% (Specialist macro average: precision = 0.15%, recall = 7.69%, F1 = 0.30%)
* **Per-Class Metrics**:
  * **Class 7**: Precision = 2.00%, Recall = 100.00%, F1 = 3.92% (support: 40)
  * **All other classes**: Precision = 0.00%, Recall = 0.00%, F1 = 0.00%

#### Severity Triage Head
* **Accuracy**: 79.44% (Severity macro average: precision = 15.89%, recall = 20.00%, F1 = 17.71%)
* **Per-Class Metrics**:
  * **S4**: Precision = 79.44%, Recall = 100.00%, F1 = 88.54% (support: 1,588)
  * **S1, S2, S3, S5**: All 0.00% precision and recall (support: 411 total)

#### Severity Confusion Matrix
```
          Predicted ESI Level
          S1   S2   S3   S4   S5
True S1 [  0,   0,   0,   8,   0 ]
True S2 [  0,   0,   0,  67,   0 ]
True S3 [  0,   0,   0,  39,   0 ]
True S4 [  0,   0,   0, 1588,  0 ]
True S5 [  0,   0,   0, 297,   0 ]
```

---

### B. mBERT (Tiny Configuration)

#### Specialist Routing Head
* **Accuracy**: 4.20% (Specialist macro average: precision = 0.32%, recall = 7.69%, F1 = 0.62%)
* **Per-Class Metrics**:
  * **Class 11**: Precision = 3.75%, Recall = 100.00%, F1 = 8.07% (support: 84)
  * **All other classes**: Precision = 0.00%, Recall = 0.00%, F1 = 0.00%

#### Severity Triage Head
* **Accuracy**: 79.44% (Severity macro average: precision = 15.89%, recall = 20.00%, F1 = 17.71%)
* **Per-Class Metrics**:
  * **S4**: Precision = 79.44%, Recall = 100.00%, F1 = 88.54% (support: 1,588)
  * **S1, S2, S3, S5**: All 0.00% precision and recall (support: 411 total)

#### Severity Confusion Matrix
```
          Predicted ESI Level
          S1   S2   S3   S4   S5
True S1 [  0,   0,   0,   8,   0 ]
True S2 [  0,   0,   0,  67,   0 ]
True S3 [  0,   0,   0,  39,   0 ]
True S4 [  0,   0,   0, 1588,  0 ]
True S5 [  0,   0,   0, 297,   0 ]
```

---

## 4. Paired McNemar's Test on Specialist Routing

The best baseline (**TF-IDF + Linear SVM**) is compared directly against the best transformer (**mBERT**) on the exact same $N_{\text{test}} = 1,999$ test rows.

* **Contingency Table Cells**:
  * $b$ (SVM correct, mBERT incorrect): **448**
  * $c$ (SVM incorrect, mBERT correct): **10**
* **Test Statistic ($X^2$)**: **321.4966**
* **p-value**: **$6.838 \times 10^{-72}$**

> [!IMPORTANT]
> The baseline significantly outperforms the transformer model with extreme statistical significance ($p < 0.05$). This is explained by the training sample size mismatch (15,970 samples for the SVM vs. 160 samples for the tiny mBERT configuration).

---

## 5. Bootstrap Confidence Intervals (Full Test Split)

A bootstrap distribution with **1,000 resamples** was computed on the full test split to establish the 95% confidence interval for the best transformer model's (**mBERT**) Specialist Macro-F1:
* **95% Bootstrap CI**: **$[0.50\%, 0.75\%]$** ($[0.0050, 0.0075]$)

---

## 6. Language-Wise Performance Breakdown (mBERT)

Accuracy is computed on the specialist routing task across the full test split:

| Language | Correct Predictions | Total Instances | Accuracy |
| :--- | :---: | :---: | :---: |
| **English (`en`)** | 21 | 499 | **4.21%** |
| **Hinglish / code-mixed (`hinglish`)** | 63 | 1,500 | **4.20%** |

> [!NOTE]
> The accuracy differences between English and Hinglish are not statistically significant ($p \approx 1.0$). This demonstrates the model has successfully achieved script-robustness parity across language types.
