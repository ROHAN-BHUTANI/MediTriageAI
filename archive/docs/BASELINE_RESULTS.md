# MediTriageAI — Baseline Evaluation Results

This document presents the metrics and confusion matrices for the three baseline models evaluated on the frozen dataset split (15,970 training rows, 1,999 test rows).

---

## 1. Performance Comparison Table

The table below summarizes the performance of the three TF-IDF models across the Specialist Routing (13 classes) and Severity Triage (5 classes) tasks.

### Specialist Routing (13 classes)
| Model | Accuracy | Macro-F1 |
| :--- | :---: | :---: |
| **TF-IDF + Logistic Regression** | 30.27% | 10.40% |
| **TF-IDF + Linear SVM** | 26.11% | 11.01% |
| **TF-IDF + Random Forest** | 25.11% | 8.26% |

### Severity Triage (5 classes)
| Model | Accuracy | Macro-F1 | Adjacent Confusion Rate | Distant Confusion Rate |
| :--- | :---: | :---: | :---: | :---: |
| **TF-IDF + Logistic Regression** | 94.40% | 63.18% | 3.95% | 1.65% |
| **TF-IDF + Linear SVM** | 97.25% | 92.61% | 1.80% | 0.95% |
| **TF-IDF + Random Forest** | 98.25% | 93.70% | 0.80% | 0.95% |

*Note*: Severity triage baselines achieve very high accuracy because the ground-truth target is derived from regular-expression heuristics. TF-IDF bag-of-words features easily latch onto the exact keyword sequences defined in the rule set, making this task trivial for simple classifiers, whereas the 13-class specialist routing represents the actual semantic generalization challenge.

---

## 2. Transformer Advantage over TF-IDF Baselines

A multilingual transformer (such as XLM-RoBERTa) is theoretically and empirically expected to outperform these TF-IDF baselines due to two core capabilities:
1. **Semantic Sequence Encoding vs. Bag-of-Words**: TF-IDF completely ignores word order, grammar, and syntactic qualifiers (e.g., negations like *"no chest pain"* vs. *"severe chest pain"*). A transformer’s self-attention mechanism processes context bidirectionally, allowing it to resolve terms like *"bleeding"* to different departments (e.g., `OBGYN` vs. `SURGERY`) based on syntactic context.
2. **Cross-Script Phonetic Robustness**: TF-IDF relies on exact token matching and fails to generalize when Hinglish transliterations introduce spelling variations (e.g., *"daktar"*, *"medisin"*, *"nhi"*, *"bohot"*). A multilingual PLM trained on code-mixed and transliterated datasets projects these phonetic variants into a shared vector space, mapping transliterations directly to their semantic anchors (e.g., mapping *"dard"* and *"pain"* to the same vector vicinity). This prevents the high out-of-vocabulary rate and classification failures that decimate baseline performance on code-mixed queries.

---

## 3. Detailed Per-Class Reports

### A. TF-IDF + Logistic Regression

#### Specialist Routing
```
               precision    recall  f1-score   support
  CARDIO_PULM       0.14      0.13      0.13       144
           ED       0.00      0.00      0.00        20
ENT_OPHTHALMO       0.00      0.00      0.00        92
      GEN_MED       0.49      0.65      0.56       628
           GI       0.12      0.07      0.09       108
        NEURO       0.09      0.06      0.08       128
        OBGYN       0.12      0.05      0.07        88
ONCOLOGY_HEME       0.00      0.00      0.00        40
        ORTHO       0.14      0.13      0.13       180
         PEDS       0.00      0.00      0.00        24
        PSYCH       0.00      0.00      0.00        16
    RENAL_URO       0.03      0.02      0.03        84
      SURGERY       0.25      0.30      0.27       447
```

#### Severity Triage
```
              precision    recall  f1-score   support
          S1       0.00      0.00      0.00         8
          S2       1.00      0.66      0.79        67
          S3       1.00      0.33      0.50        39
          S4       0.94      0.99      0.97      1588
          S5       0.94      0.86      0.90       297
```

### B. TF-IDF + Linear SVM

#### Specialist Routing
```
               precision    recall  f1-score   support
  CARDIO_PULM       0.13      0.12      0.12       144
           ED       0.00      0.00      0.00        20
ENT_OPHTHALMO       0.00      0.00      0.00        92
      GEN_MED       0.46      0.54      0.50       628
           GI       0.08      0.06      0.06       108
        NEURO       0.11      0.11      0.11       128
        OBGYN       0.21      0.13      0.16        88
ONCOLOGY_HEME       0.03      0.03      0.03        40
        ORTHO       0.16      0.17      0.16       180
         PEDS       0.00      0.00      0.00        24
        PSYCH       0.00      0.00      0.00        16
    RENAL_URO       0.08      0.07      0.08        84
      SURGERY       0.20      0.22      0.21       447
```

#### Severity Triage
```
              precision    recall  f1-score   support
          S1       1.00      1.00      1.00         8
          S2       0.98      0.76      0.86        67
          S3       0.86      0.82      0.84        39
          S4       0.98      0.99      0.98      1588
          S5       0.94      0.95      0.95       297
```

### C. TF-IDF + Random Forest

#### Specialist Routing
```
               precision    recall  f1-score   support
  CARDIO_PULM       0.09      0.08      0.09       144
           ED       0.00      0.00      0.00        20
ENT_OPHTHALMO       0.00      0.00      0.00        92
      GEN_MED       0.46      0.52      0.49       628
           GI       0.00      0.00      0.00       108
        NEURO       0.13      0.13      0.13       128
        OBGYN       0.00      0.00      0.00        88
ONCOLOGY_HEME       0.00      0.00      0.00        40
        ORTHO       0.12      0.12      0.12       180
         PEDS       0.00      0.00      0.00        24
        PSYCH       0.00      0.00      0.00        16
    RENAL_URO       0.00      0.00      0.00        84
      SURGERY       0.23      0.28      0.26       447
```

#### Severity Triage
```
              precision    recall  f1-score   support
          S1       1.00      1.00      1.00         8
          S2       0.98      0.76      0.86        67
          S3       0.89      0.82      0.85        39
          S4       0.98      1.00      0.99      1588
          S5       1.00      0.97      0.98       297
```

---

## 4. Confusion Matrices (Visualizations & Raw Matrices)

### Logistic Regression
#### Specialist Confusion Matrix
![Logistic Regression Specialist Confusion Matrix](file:///C:/Users/bhuta/.gemini/antigravity-ide/brain/56c6fc72-8834-4113-8a29-1dc288e32568/lr_specialist_cm.png)

#### Severity Confusion Matrix
![Logistic Regression Severity Confusion Matrix](file:///C:/Users/bhuta/.gemini/antigravity-ide/brain/56c6fc72-8834-4113-8a29-1dc288e32568/lr_severity_cm.png)

Raw Severity Matrix:
```
[0,  0,  0,    8,   0]  # S1 (True)
[0, 44,  0,   23,   0]  # S2 (True)
[0,  0, 13,   24,   2]  # S3 (True)
[0,  0,  0, 1574,  14]  # S4 (True)
[0,  0,  0,   41, 256]  # S5 (True)
```

---

### Linear SVM
#### Specialist Confusion Matrix
![Linear SVM Specialist Confusion Matrix](file:///C:/Users/bhuta/.gemini/antigravity-ide/brain/56c6fc72-8834-4113-8a29-1dc288e32568/svm_specialist_cm.png)

#### Severity Confusion Matrix
![Linear SVM Severity Confusion Matrix](file:///C:/Users/bhuta/.gemini/antigravity-ide/brain/56c6fc72-8834-4113-8a29-1dc288e32568/svm_severity_cm.png)

Raw Severity Matrix:
```
[8,  0,  0,    0,   0]  # S1 (True)
[0, 51,  0,   12,   4]  # S2 (True)
[0,  0, 32,    7,   0]  # S3 (True)
[0,  1,  3, 1570,  14]  # S4 (True)
[0,  0,  2,   12, 283]  # S5 (True)
```

---

### Random Forest
#### Specialist Confusion Matrix
![Random Forest Specialist Confusion Matrix](file:///C:/Users/bhuta/.gemini/antigravity-ide/brain/56c6fc72-8834-4113-8a29-1dc288e32568/rf_specialist_cm.png)

#### Severity Confusion Matrix
![Random Forest Severity Confusion Matrix](file:///C:/Users/bhuta/.gemini/antigravity-ide/brain/56c6fc72-8834-4113-8a29-1dc288e32568/rf_severity_cm.png)

Raw Severity Matrix:
```
[8,  0,  0,    0,   0]  # S1 (True)
[0, 51,  0,   16,   0]  # S2 (True)
[0,  0, 32,    7,   0]  # S3 (True)
[0,  1,  2, 1584,   1]  # S4 (True)
[0,  0,  2,    6, 289]  # S5 (True)
```
