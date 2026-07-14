# MediTriageAI — Model Retraining Zoo Log ($N_{\text{train}} = 3,000$)

This log captures the training time, validation curves, and convergence diagnostics for the scaled-up training split ($N_{\text{train}} = 3,000$) on CPU hardware.

---

## 1. Retraining Results Summary

| Model | Training Time (s) | Epoch 1 Train Loss | Epoch 2 Train Loss | Epoch 1 Val Loss | Epoch 2 Val Loss | Specialist Macro-F1 (Test) | Severity Macro-F1 (Test) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **mBERT** | 546.33s | 1.8843 | 0.9990 | 4.5744 | 5.3159 | **2.81%** | **17.71%** |
| **DistilBERT-multi** | 632.05s | 1.9283 | 1.0187 | 4.4832 | 5.1529 | **2.81%** | **17.71%** |

---

## 2. Loss Curves & Convergence Diagnosis

### mBERT Loss Curves
* **Train Loss**: Decreases from **1.8843** (Epoch 1) to **0.9990** (Epoch 2).
* **Val Loss**: Increases from **4.5744** (Epoch 1) to **5.3159** (Epoch 2).

### DistilBERT-multilingual Loss Curves
* **Train Loss**: Decreases from **1.9283** (Epoch 1) to **1.0187** (Epoch 2).
* **Val Loss**: Increases from **4.4832** (Epoch 1) to **5.1529** (Epoch 2).

### Convergence Verdict
> [!WARNING]
> **DIAGNOSIS: SEVERE OVERFITTING**
> The training loss decreases sharply for both models (dropping by ~47% in one epoch), while the validation loss rises (increasing by 16% for mBERT and 15% for DistilBERT). This divergent behavior is a classic signature of aggressive overfitting.
> 
> While training accuracy on the 3,000-sample subset reached **80.60% (Specialist)** and **93.57% (Severity)**, the validation accuracy remained stuck at **21.72% (Specialist)** and **81.39% (Severity)**. On the unseen test set, the models collapsed entirely back to predicting a single majority class (Class 12 for Specialist, ESI S4 for Severity), yielding near-chance F1 scores.

---

## 3. Why the Performance Collapse Persists

This collapse is a direct consequence of the resource-constrained simulation settings:
1. **Tiny Model Capacity**: `ZooConfig` restricts the transformers to 2 layers and a hidden size of 64 to allow local CPU training. 
2. **From-Scratch Training**: The models are randomly initialized and trained from scratch rather than using pretrained weights (since Hugging Face remote downloading is disabled in the offline evaluation sandbox). A tiny transformer trained from scratch on 3,000 samples cannot learn generalizable multi-class clinical triage representations and simply memorizes the training data.
3. **Hyperparameter Mismatch**: The learning rates ($2\times 10^{-5}$ encoder, $1\times 10^{-4}$ heads) are designed for fine-tuning large pre-trained models. For scratch training of a 64-dimensional model, these rates lead to rapid overfitting.
