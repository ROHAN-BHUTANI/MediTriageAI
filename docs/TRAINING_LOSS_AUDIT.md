# MediTriageAI — Loss Function and Masking Audit

**Specification Baseline:** `v1.0.0-FROZEN`  
**Audit Date:** `2026-08-16`  
**Inspected Implementation:** [src/model.py:75-150](file:///c:/Users/bhuta/Desktop/MediTriageAI_Data_Engine/src/model.py#L75-L150) (`FocalLoss`, `JointLoss`)

---

## 1. Multi-Task Joint Loss Formulation

The loss is a linear combination of specialist routing Focal Loss and triage severity Focal Loss:

$$\mathcal{L}_{\text{joint}} = \alpha \cdot \mathcal{L}_{\text{specialist}} + \beta \cdot \mathcal{L}_{\text{severity}}$$

where:
- $\alpha = 1.0$ (Specialist routing weight)
- $\beta = 1.2$ (Severity acuity weight)
- $\gamma = 2.0$ (Focal focusing parameter)
- $\text{ignore\_index} = -1$

### Mathematical Definition of Masked Focal Loss:
For input logits $z \in \mathbb{R}^{C}$ and target label $y \in \{-1, 0, \dots, C-1\}$:
$$p_t = \frac{\exp(z_y)}{\sum_j \exp(z_j)}$$
$$\text{FL}(p_t) = -(1 - p_t)^\gamma \log(p_t) \cdot \mathbb{I}(y \neq -1)$$

When $y = -1$, the loss is strictly $0.0$, and the sample does not contribute to the reduction denominator:
$$\mathcal{L} = \frac{\sum_{i=1}^B \text{FL}(p_{t, i}) \cdot \mathbb{I}(y_i \neq -1)}{\max\left(1, \sum_{i=1}^B \mathbb{I}(y_i \neq -1)\right)}$$

---

## 2. Empirical Loss Function Test Matrix

| Test Scenario | Specialist Labels | Severity Labels | Specialist Loss | Severity Loss | Joint Loss | NaN / Inf? |
|---|---|---|---|---|---|---|
| **Standard Mixed Batch** | Random 0..12 | Random 0..4 | 2.2154 | 1.2538 | 3.7200 | **False (Clean)** |
| **All Unlabeled Severity** | Random 0..12 | All `-1` | 2.2873 | 0.0000 | 2.2873 | **False (Clean)** |
| **All Unlabeled Specialist** | All `-1` | Random 0..4 | 0.0000 | 1.4120 | 1.6944 | **False (Clean)** |
| **All Unlabeled Both Tasks** | All `-1` | All `-1` | 0.0000 | 0.0000 | 0.0000 | **False (Clean)** |
| **Extreme Logits ($\pm 100$)** | Random 0..12 | Random 0..4 | Stable | Stable | Stable | **False (Clean)** |

---

## 3. NaN Stability Invariant Verification

1. **Zero Valid Targets in Batch:**
   When a batch contains exclusively records from datasets without ESI triage levels (such as MTSamples or NEISS), $\sum \mathbb{I}(y_i \neq -1) = 0$.
   The denominator is clamped: `num_valid.clamp(min=1) == 1.0`.
   The numerator is `0.0`.
   Therefore, $\mathcal{L}_{\text{severity}} = 0.0 / 1.0 = 0.0$ (and does NOT evaluate to `0 / 0 = NaN`).
2. **Gradient Backward Pass:**
   Backpropagation through masked samples generates exactly $0.0$ gradient on the severity classification head while fully updating the encoder and specialist classification head.
