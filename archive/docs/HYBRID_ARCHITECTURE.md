# Hybrid Triage Architecture Design

## Overview
Purely automated machine learning pipelines carry substantial clinical risk when deployed in isolation, particularly for critical triage workflows where misclassification of an emergency (e.g., stroke, myocardial infarction) can result in severe adverse patient outcomes. 

To mitigate this, MediTriageAI employs a **Hybrid Triage Pipeline** that combines deterministic, rule-based heuristics with probabilistic ML predictions.

## Design Rationale

### 1. Rule-Based Red-Flag Layer (Safety First)
Machine learning models, particularly Transformers, can hallucinate or fail silently on out-of-distribution texts. To prevent catastrophic failure on clear emergencies:
- **Implementation:** A predefined list of high-risk keywords (e.g., `"chest pain"`, `"radiation"`, `"loss of consciousness"`, `"severe bleeding"`, `"stroke"`, `"suicide"`) is continuously evaluated against the patient text.
- **Action:** If any red flag is triggered, the system explicitly flags the case (`requires_manual_review = True`), allowing clinical staff to intercept the request immediately regardless of what the ML model predicts.

### 2. Top-3 Specialist Shortlist (Human-in-the-Loop Efficiency)
In the initial baseline analysis, we observed significant taxonomic overlap (e.g., Surgery vs. Orthopedics, General Medicine vs. Cardiopulmonary). Forcing a model to choose a single, mutually exclusive category leads to artificial errors and reduces trust.
- **Implementation:** The model now outputs a ranked shortlist of the top 3 predicted specialists along with their respective confidence scores.
- **Action:** Instead of blindly routing a patient to one department, human triage nurses are presented with a shortlist. This dramatically reduces cognitive load while maintaining human oversight for the final decision. 

### 3. Explicit Low-Confidence Escalation (Uncertainty Awareness)
Neural networks can sometimes output uniform probability distributions when they are confused by ambiguous or multi-system complaints.
- **Implementation:** If the top predicted specialist has a confidence score below a strict threshold (`< 0.60`), the system triggers an escalation.
- **Action:** The case is flagged (`requires_manual_review = True`), ensuring that edge cases or highly complex patients are not automatically routed with low certainty.

## Conclusion
This hybrid architecture bridges the gap between the scalability of AI and the stringent safety requirements of clinical triage, ensuring that critical emergencies and ambiguous cases are always escalated to human experts.
