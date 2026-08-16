# MediTriageAI — Training Evaluation Contract and Metric Protocol

**Specification Baseline:** `v1.0.0-FROZEN`  
**Document Status:** AUTHORITATIVE METRIC SPECIFICATION  
**Date:** `2026-08-16`

---

## 1. Primary and Secondary Evaluation Metrics

All model evaluations executed via [scripts/evaluate.py](file:///c:/Users/bhuta/Desktop/MediTriageAI_Data_Engine/scripts/evaluate.py) or test harnesses must report the following metrics without retrospective metric selection:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          PRIMARY BENCHMARK METRICS                          │
├────────────────────────┬────────────────────────────────────────────────────┤
│ Specialist Routing     │ Macro-F1 (13 classes), Top-1 Accuracy, Top-3 Acc,  │
│                        │ Per-Class F1, Macro AUROC                          │
├────────────────────────┼────────────────────────────────────────────────────┤
│ ESI Triage Severity    │ Macro-F1 (5 classes), Quadratic Weighted Kappa,    │
│                        │ Mean Absolute Error (MAE), Expected Calibration    │
│                        │ Error (ECE), Multi-class Brier Score               │
└────────────────────────┴────────────────────────────────────────────────────┘
```

---

## 2. Mandatory Subgroup Slicing Dimensions

In addition to aggregate test metrics, the evaluation contract mandates granular breakdown across 6 clinical and linguistic axes:

1. **Linguistic Slices:**
   - English Baseline (`en`)
   - Hinglish Code-Mixed (`hi-en`)
   - Devanagari Hindi (`hi`)
   - Romanized Hindi (`hi-Latn`)
2. **Robustness & Perturbation Strata:**
   - Shorthand / Abbreviated Notation
   - Lexical & Clinical Synonyms
   - ASR Noise & Typos
   - Colloquial Indian English
   - Late Red Flag Clinical Escalation
   - Clinical Hard Negatives
3. **Department Slices:**
   - All 13 individual specialist classes (specifically evaluating `OBGYN`, `PEDS`, `ONCOLOGY_HEME`, `PSYCH`).
4. **Severity Slices:**
   - All 5 ESI levels (specifically monitoring $S1$ under-triage rate $\Pr(\hat{y} \geq S3 \mid y = S1)$).
5. **Statistical Confidence Intervals:**
   - 1,000 bootstrap resamples on the test set to report empirical 95% Confidence Intervals for Macro-F1.

---

## 3. Evaluation Output Envelope Specification (`evaluation_report.json`)

```json
{
  "version": "1.0.0",
  "checkpoint_path": "results/xlm_roberta_large/checkpoint.pt",
  "dataset_manifest_hash": "f64ed360b246416cf3b117a27f9c09843f1ad53430a3fd2575358587c1902513",
  "test_samples_total": 5355,
  "test_samples_severity_labeled": 2049,
  "metrics": {
    "specialist": {
      "macro_f1": 0.0,
      "top1_accuracy": 0.0,
      "top3_accuracy": 0.0,
      "per_class_f1": {},
      "bootstrap_95_ci": [0.0, 0.0]
    },
    "severity": {
      "macro_f1": 0.0,
      "quadratic_weighted_kappa": 0.0,
      "mae": 0.0,
      "ece": 0.0,
      "brier_score": 0.0,
      "per_class_f1": {},
      "bootstrap_95_ci": [0.0, 0.0]
    }
  },
  "subgroup_analysis": {
    "language": {},
    "robustness_strata": {},
    "department": {},
    "severity": {}
  }
}
```
