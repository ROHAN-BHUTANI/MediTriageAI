# Specialist Routing Confusion Analysis

This analysis examines the classical SVM baseline's predictions on the full 1,999-row test set for the Specialist Routing task to identify root causes of classification failure, specifically distinguishing between genuine clinical ambiguity and label noise.

## 1. Full 13x13 Confusion Matrix

Labels (in order): `CARDIO_PULM`, `ED`, `ENT_OPHTHALMO`, `GEN_MED`, `GI`, `NEURO`, `OBGYN`, `ONCOLOGY_HEME`, `ORTHO`, `PEDS`, `PSYCH`, `RENAL_URO`, `SURGERY`.

```text
[
  [18,  0,  4, 66,  1,  4,  0,  0,  4,  0,  0,  0, 47], # CARDIO_PULM
  [ 0,  0,  0, 12,  0,  0,  4,  0,  4,  0,  0,  0,  0], # ED
  [ 0,  0,  0, 24,  0,  0,  0,  4,  0,  0,  0,  0, 64], # ENT_OPHTHALMO
  [46, 18, 26,329,  6, 69, 20, 24, 61,  5,  0, 15,  9], # GEN_MED
  [ 0,  4,  0, 33,  8,  2,  0,  0,  0,  0,  0,  5, 56], # GI
  [ 4,  0,  0, 66,  0, 11,  0,  0, 28,  0,  0,  0, 19], # NEURO
  [ 0,  0,  0, 12,  0,  0, 19,  0,  0,  0,  0,  0, 57], # OBGYN
  [ 0,  0,  0, 23,  0,  0,  0,  1,  0,  0,  0,  0, 16], # ONCOLOGY_HEME
  [ 0,  0,  0, 56,  0, 24,  0,  0, 35,  0,  0,  0, 65], # ORTHO
  [ 0,  0,  0, 13,  2,  1,  0,  0,  0,  0,  0,  3,  5], # PEDS
  [ 0,  0,  0, 15,  0,  0,  0,  0,  0,  0,  1,  0,  0], # PSYCH
  [ 0,  0,  0, 28,  0,  0,  0,  0,  0,  0,  4,  5, 47], # RENAL_URO
  [68,  0, 33, 37, 64, 15, 17,  1, 66,  1,  0, 44,102]  # SURGERY
]
```

## 2. Top 5 Confusion Pairs & Case Analysis

### Pair 1: GEN_MED -> NEURO (69 times)
**Example text snippet:** *"The patient referred by Dr. X for evaluation of her possible tethered cord... who underwent a lipomyomeningocele repair... leg pain... Possible tethered cord."*
**Analysis:** This is **Label Noise / Bad Ground Truth**. A tethered cord and lipomyomeningocele repair is unequivocally a Neurology/Neurosurgery case, yet it is labeled as `GEN_MED` in the dataset. The model correctly identifies it as `NEURO`, but gets penalized.

### Pair 2: SURGERY -> CARDIO_PULM (68 times)
**Example text snippet:** *"PREOPERATIVE DIAGNOSES: 1. Oxygen dependency. 2. Chronic obstructive pulmonary disease. PROCEDURES PERFORMED: 1. Tracheostomy with skin flaps. 2. SCOOP procedure FastTract."*
**Analysis:** This is **Genuinely Ambiguous / Overlapping**. The text describes a surgical procedure (Tracheostomy) for a pulmonary condition (COPD). It is labeled `SURGERY`, but `CARDIO_PULM` (Cardiothoracic Surgery / Pulmonology) is an equally valid, if not more precise, classification.

### Pair 3: CARDIO_PULM -> GEN_MED (66 times)
**Example text snippet:** *"This is a 61-year-old woman with a history of polyarteritis nodosa, mononeuritis multiplex... severe sleep apnea... evaluating for difficulty in initiating and maintaining sleep... Severe central sleep apnea."*
**Analysis:** This is **Genuinely Ambiguous / Overlapping**. Sleep apnea and hypoxemia can be managed by Pulmonologists (`CARDIO_PULM`) or Internal Medicine (`GEN_MED`) physicians. The text contains general medical history intertwined with a pulmonology focus.

### Pair 4: NEURO -> GEN_MED (66 times)
**Example text snippet:** *"This is an 83-year-old woman referred for diagnostic lumbar puncture for possible malignancy... presumed non-small cell lung cancer... stopped walking... left arm has become gradually less functional."*
**Analysis:** This is **Genuinely Ambiguous**. The primary procedure is a lumbar puncture (Neurology), but the underlying context is lung cancer, generalized weakness, and loss of appetite, which are strong `GEN_MED` or `ONCOLOGY_HEME` signals. The TF-IDF model gets confused by the broad medical context.

### Pair 5: SURGERY -> ORTHO (66 times)
**Example text snippet:** *"PREOPERATIVE DIAGNOSIS: 1. Right cubital tunnel syndrome. 2. Right carpal tunnel syndrome... PROCEDURES: 1. Right ulnar nerve transposition. 2. Right carpal tunnel release."*
**Analysis:** This is **Taxonomic Overlap**. Carpal tunnel release is an orthopedic/hand surgery. The dataset has separate classes for `SURGERY` and `ORTHO`, but Orthopedics is a surgical subspecialty. Classifying this as `ORTHO` when the ground truth says `SURGERY` is not clinically wrong; it's just a taxonomy clash.

---

## 3. Proposed Consolidated Taxonomy

Based on the evidence, the poor performance of the baseline (and transformers) is heavily exacerbated by an overlapping, poorly separated taxonomy and noisy ground truth labels. 

**Recommendation: Consolidate the 13 categories into 4-5 clinically coherent Supergroups.**

1. **SURGICAL & ORTHOPEDIC SCIENCES** 
   - *Merges:* `SURGERY`, `ORTHO`, and surgical cases within `ENT_OPHTHALMO` / `OBGYN`.
   - *Rationale:* The models constantly confuse `SURGERY` with `ORTHO` (and other procedural texts). A carpal tunnel release is both surgery and ortho; forcing the model to choose one introduces artificial errors.
2. **INTERNAL MEDICINE & CARDIOPULMONARY**
   - *Merges:* `GEN_MED`, `CARDIO_PULM`, `GI`, `RENAL_URO`, `ONCOLOGY_HEME`.
   - *Rationale:* `GEN_MED` acts as a massive sinkhole for the dataset (causing the label noise seen in Pair 1). `CARDIO_PULM` and `GEN_MED` are heavily confused due to shared vocabulary (e.g., sleep apnea, breathing issues). 
3. **NEUROLOGY & PSYCHIATRY**
   - *Merges:* `NEURO`, `PSYCH`.
   - *Rationale:* Shared focus on the central nervous system and behavioral symptoms.
4. **EMERGENCY & TRAUMA**
   - *Keeps:* `ED`. 
5. **PEDIATRICS & WOMEN'S HEALTH** (Optional depending on data)
   - *Merges:* `PEDS`, `OBGYN`.

**Conclusion:** 
Without consolidating the taxonomy (or cleaning the ground truth where `NEURO` cases are labeled `GEN_MED`), no model—whether a simple SVM or a massive Transformer—will achieve high accuracy. The dataset's current labels force models to guess arbitrary taxonomy distinctions rather than learn underlying clinical semantics.
