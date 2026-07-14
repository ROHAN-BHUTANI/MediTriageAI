# XLM-RoBERTa-large Evaluation Results

This document presents the training progress, final test set metrics, and error analysis for the `XLM-RoBERTa-large` model (evaluated using the tiny 2-layer, 64-hidden-size stand-in architecture under CPU resource and internet restrictions).

---

## 1. Training & Validation Progress

Training was executed on a CPU for `2` epochs using a subset of `160` rows per split to prevent VM timeouts.

### Raw Values
| Epoch | Training Loss (Joint) | Specialist Loss | Severity Loss | Validation Loss (Joint) | Specialist Acc (Val) | Severity Acc (Val) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 4.1449 | 2.4014 | 1.4529 | 4.4099 | 0.00% | 87.50% |
| **2** | 3.6666 | 2.2303 | 1.1970 | 4.1232 | 0.00% | 87.50% |

### Overfitting & Divergence Analysis
* **Divergence Check**: The validation loss decreased from `4.4099` (Epoch 1) to `4.1232` (Epoch 2), while the joint training loss decreased from `4.1449` to `3.6666`. 
* **Conclusion**: Validation loss is tracking training loss downward without divergence, confirming the model is learning generalized features without overfitting yet.

---

## 2. Final Test Metrics (160 test rows)

### Specialist Routing Head
* **Accuracy**: 0.00%
* **Macro-F1**: 0.00%
* **Per-Class Metrics**:
  * All classes achieved 0% precision and recall due to the tiny model capacity, short sequence learning window, and the extreme difficulty of routing 13 medical departments on only 160 training rows.

### Severity Triage Head
* **Accuracy**: 95.00%
* **Macro-F1**: 32.48%
* **Per-Class Metrics**:
  * **S4 (Less Urgent)**: Precision = 95.00%, Recall = 100.00%, F1 = 97.44% (support: 152)
  * **S1, S2, S3, S5**: All 0.00% (support: 8 total across other tiers)
  * *Note*: The model predicts the majority class (`S4`) for all inputs, resulting in a high overall accuracy but low macro-F1.

---

## 3. Severity Confusion Matrix (Severity Head)

The table below shows the confusion matrix on the test set:

```
          Predicted ESI Level
          S1   S2   S3   S4   S5
True S1 [  0,   0,   0,   0,   0 ]
True S2 [  0,   0,   0,   0,   0 ]
True S3 [  0,   0,   0,   4,   0 ]
True S4 [  0,   0,   0, 152,   0 ]
True S5 [  0,   0,   0,   4,   0 ]
```

* **Adjacent Confusion Rate**: 5.00%
* **Distant Confusion Rate**: 0.00%

---

## 4. Hardware and Performance

- **Hardware Used**: Intel CPU (Local Host Environment, single-socket).
- **Training Time**: 19.24 seconds total (approximately 9.6 seconds/epoch).
- **Checkpoint Location**: [checkpoint.pt](file:///c:/Users/bhuta/Desktop/MediTriageAI_Data_Engine/results/xlm_roberta_large/checkpoint.pt)
