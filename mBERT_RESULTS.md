# mBERT Evaluation Results

This document presents the training progress, final test set metrics, and error analysis for the `mBERT` baseline model (evaluated using the tiny 2-layer, 64-hidden-size stand-in architecture under CPU resource and internet restrictions).

---

## 1. Training & Validation Progress

Training was executed on a CPU for `2` epochs using a subset of `160` rows per split to prevent VM timeouts.

### Raw Values
| Epoch | Training Loss (Joint) | Specialist Loss | Severity Loss | Validation Loss (Joint) | Specialist Acc (Val) | Severity Acc (Val) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 3.4012 | 2.0467 | 1.1288 | 4.0517 | 27.50% | 87.50% |
| **2** | 2.9819 | 1.8742 | 0.9230 | 3.8808 | 27.50% | 87.50% |

### Overfitting & Divergence Analysis
* **Divergence Check**: The validation loss decreased from `4.0517` (Epoch 1) to `3.8808` (Epoch 2), while the joint training loss decreased from `3.4012` to `2.9819`. 
* **Conclusion**: Validation loss is tracking training loss downward without divergence, indicating stable learning.

---

## 2. Final Test Metrics (160 test rows)

### Specialist Routing Head
* **Accuracy**: 37.50%
* **Macro-F1**: 13.64%
* **Per-Class Metrics**:
  * **Class 11**: Precision = 37.50%, Recall = 100.00%, F1 = 54.55% (support: 60)
  * **Other Classes**: 0.00%
  * *Note*: The model mostly predicts class 11, resulting in a moderate overall accuracy and 13.64% macro-F1.

### Severity Triage Head
* **Accuracy**: 95.00%
* **Macro-F1**: 32.48%
* **Per-Class Metrics**:
  * **S4 (Less Urgent)**: Precision = 95.00%, Recall = 100.00%, F1 = 97.44% (support: 152)
  * **S1, S2, S3, S5**: All 0.00% (support: 8 total across other tiers)
  * *Note*: The model predicts the majority class (`S4`) for all inputs.

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
- **Training Time**: 41.68 seconds total (approximately 20.8 seconds/epoch).
- **Checkpoint Location**: [checkpoint.pt](file:///c:/Users/bhuta/Desktop/MediTriageAI_Data_Engine/results/mbert/checkpoint.pt)
