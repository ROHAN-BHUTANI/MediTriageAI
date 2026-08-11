# MediTriageAI — Dataset Governance

> **Version**: 1.0.0  
> **Last Updated**: 2026-08-11  
> **Branch**: `training-pipeline`  

---

## Purpose

This document defines the official production dataset governance policy for MediTriageAI. It classifies every dataset by its role (CORE, AUXILIARY, or EXCLUDED) and documents known risks, limitations, and quality considerations.

> **Important**: CORE / AUXILIARY classification is a governance recommendation reflecting data relevance and label quality. It is NOT a claim that any dataset has perfect labels or zero risks.

---

## Production Dataset Classification

### CORE Datasets

CORE datasets provide the highest-quality, most clinically relevant training signal for the MediTriageAI triage routing and severity prediction objectives.

| Dataset | Rows (approx.) | Specialist Label | Severity Label | Justification |
|:---|:---:|:---|:---|:---|
| `neiss` | 7,326,429 | MAPPED (diagnosis/body codes) | NONE | Largest dataset. Real CPSC ED injury narratives with structured diagnosis and body-part codes for specialist routing. |
| `nhamcs_ed` | 50,548 | DIRECT (ED only) | DIRECT (ESI 1-5) | Only source of real government-validated Emergency Severity Index (ESI) triage scores from CDC survey data. |
| `mtsamples` | 4,999 | MAPPED (curated) | NONE | Highest-quality specialist labels via curated `RAW_TO_DEPARTMENT` mapping from real medical transcriptions. |

### AUXILIARY Datasets

AUXILIARY datasets supplement CORE data with additional medical language coverage, specialty routing signals, or secondary triage information. All carry documented quality caveats.

| Dataset | Rows (approx.) | Specialist Label | Severity Label | Justification |
|:---|:---:|:---|:---|:---|
| `pmc_patients` | 167,034 | INFERRED (keyword) | NONE | Rich clinical case summaries from published literature. Keyword-inferred specialist routing. |
| `chatdoctor_healthcaremagic` | 112,156 | INFERRED (keyword) | NONE | Patient question format approximates clinical presentation. Label leakage from doctor answers. |
| `chatdoctor_icliniq` | 7,321 | INFERRED (keyword) | NONE | Supplements healthcaremagic with different patient population. Same label leakage caveat. |
| `symptom2disease` | 1,200 | MAPPED (19-disease) | NONE | Clean symptom→disease→department mapping. Small but high label precision. |
| `meddialog_en` | 2,725,990 | INFERRED (keyword) | NONE | Massive volume of doctor-patient dialog for medical language understanding. Conversational format. |
| `fedmml_ed_triage` | 87,234 | DIRECT (ED only) | DIRECT (ESI 1-5) | Structured clinical fields with ESI severity labels. **SYNTHETIC (LLM-generated)** — use with documented caution. |
| `kaggle_medical_triage` | 1,112 | DIRECT/MAPPED | DIRECT (urgency level) | Has both specialty and severity labels. Very small dataset. |

### EXCLUDED Datasets

EXCLUDED datasets are present in the repository (adapters, downloaders, metadata preserved for reproducibility) but are NOT included in the production `active_datasets` configuration.

| Dataset | Rows (approx.) | Reason for Exclusion |
|:---|:---:|:---|
| `l3cube_code_mixed` | 38,176 | **Non-medical corpus.** This is a Hindi-English language identification dataset with zero clinical content. Department labels are fabricated from Hinglish keywords unrelated to the actual sentences. Inclusion injects pure noise into training. |
| `medical_meadow_medqa` | 10,178 | **Zero usable rows.** Adapter sets both `department = None` and `triage_level = None`. Schema validation (`src/schema.py`) drops 100% of rows because they fail the supervision requirement (`valid_dept | valid_triage`). Even if labels were added, content is instruction-tuning QA format, not clinical presentation. |
| `medqa_usmle` | 11,451 | **Format mismatch.** USMLE multiple-choice question stems are medical exam items, not patient presentations. Keyword-inferred specialist labels are unreliable. Teaches MCQ answering patterns rather than triage routing. |

---

## Known Dataset Risks

### 1. Keyword-Derived Specialist Labels (HIGH)

**Affected datasets**: `pmc_patients`, `chatdoctor_healthcaremagic`, `chatdoctor_icliniq`, `meddialog_en`

Specialist department labels for these datasets are inferred by keyword-regex matching on clinical text (e.g., "cardiac" → `CARDIO_PULM`, "fracture" → `ORTHO`). This creates a circularity risk: the model learns to predict labels that were assigned by the same type of keyword detection.

**Mitigation**: Document as known limitation. Consider manual label validation on a sample in future work.

### 2. Severity Label Sparsity (HIGH)

Only 3 of 10 active datasets provide severity/triage labels:
- `nhamcs_ed`: ~50K rows with real ESI scores
- `fedmml_ed_triage`: ~87K rows with ESI labels (SYNTHETIC)
- `kaggle_medical_triage`: ~1.1K rows with urgency levels

Total severity-labeled rows: ~138K out of ~10.5M total (~1.3%). The severity prediction head trains on a small fraction of available data.

**Mitigation**: Document as known limitation. Future work should evaluate whether severity head performance is acceptable for production use.

### 3. NEISS Dataset Dominance (MEDIUM)

NEISS contributes ~7.3M of ~10.5M total rows (~70%). All NEISS records are injury-related ED narratives. This creates a training bias toward injury presentations and away from medical illness presentations.

**Mitigation**: The deduplication and split pipeline treats all datasets uniformly. Consider class-weighted sampling or dataset-proportional sampling in future training configurations.

### 4. Synthetic FedMML Data (MEDIUM)

`fedmml_ed_triage` is explicitly described as "Synthetic ED triage encounters." It provides 63% of all severity-labeled rows. The model may learn generation artifacts rather than real clinical patterns.

**Mitigation**: Document as known limitation. Consider ablation studies comparing model performance with and without FedMML.

### 5. NHAMCS Coded-Text Representation (MEDIUM)

NHAMCS text is synthesized from fixed-width survey fields: `"Age: 45, Sex: Male, Reason for Visit 1 (Code): 2010"`. This is not natural language and requires the model to learn numeric visit code interpretation.

**Mitigation**: Document as known limitation. NHAMCS provides the only real ESI scores and its structured format may actually help the model learn systematic triage reasoning.

### 6. Conversational Label Leakage (MEDIUM)

**Affected datasets**: `chatdoctor_healthcaremagic`, `chatdoctor_icliniq`

Department labels are inferred using both patient input AND doctor output text. However, `raw_text` for training contains only the patient input. The doctor's response (which names specialties explicitly) influences the label but is not available to the model at inference time.

**Mitigation**: Document as known limitation. Labels are still directionally correct (doctor responses do indicate the relevant specialty) but may be more precise than what patient text alone can support.

---

## Governance Policy

1. **Active dataset changes** require updating `config/dataset_config.yaml` AND this governance document.
2. **Excluded datasets** retain their adapters, downloaders, and metadata for reproducibility.
3. **New datasets** must be evaluated for triage relevance, label quality, and data quality risks before activation.
4. **Label modifications** must be documented in `docs/dataset_label_provenance.md`.
5. **Severity label coverage** remains a tracked limitation until explicitly addressed.
