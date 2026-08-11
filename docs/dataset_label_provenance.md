# MediTriageAI — Dataset Label Provenance

> **Version**: 1.0.0  
> **Last Updated**: 2026-08-11  
> **Branch**: `training-pipeline`  

---

## Purpose

This document records the exact provenance, type, and known limitations of every label produced by every active dataset adapter. It enables auditing of training signal quality without inspecting individual adapter source code.

---

## Label Type Definitions

| Type | Definition |
|:---|:---|
| **DIRECT** | Label comes directly from the source dataset as a first-class field (e.g., ESI triage score from CDC survey). |
| **MAPPED** | Label is translated from a source field via a curated, deterministic mapping table (e.g., MTSamples `medical_specialty` → `RAW_TO_DEPARTMENT`). |
| **INFERRED** | Label is assigned by keyword-regex matching on text content. No native label exists in the source data. |
| **NONE** | No label of this type is produced. The field is always `None`. |

---

## Per-Dataset Label Provenance

### 1. `mtsamples`

| Property | Value |
|:---|:---|
| **Adapter** | `meditriage/builder/adapters/mtsamples.py` |
| **Specialist label source** | `medical_specialty` column from source CSV |
| **Specialist label type** | **MAPPED** |
| **Specialist mapping** | `src/specialty_mapping.py` → `RAW_TO_DEPARTMENT` (71 entries, case-insensitive). Unmapped specialties fall back to `GEN_MED`. |
| **Severity label source** | — |
| **Severity label type** | **NONE** (always `None`) |
| **Known leakage risks** | LOW — Labels derive from document-level `medical_specialty` metadata, not from text content. |
| **Known limitations** | Only ~5K rows. Some specialties (e.g., "SOAP / Chart / Progress Notes") map to `GEN_MED` with low confidence. |

---

### 2. `pmc_patients`

| Property | Value |
|:---|:---|
| **Adapter** | `meditriage/builder/adapters/pmc_patients.py` |
| **Specialist label source** | Keyword-regex on `title + patient` text |
| **Specialist label type** | **INFERRED** |
| **Specialist mapping** | Hierarchical keyword taxonomy in adapter: oncology → surgery → psychiatry → ENT → urology → OB/GYN → GI → orthopedics → neurology → cardio/pulm → oncology → pediatrics (age < 18 override). Default: `GEN_MED`. |
| **Severity label source** | — |
| **Severity label type** | **NONE** (always `None`) |
| **Known leakage risks** | HIGH — Specialist keywords used for label assignment (e.g., "cardiac", "fracture") will appear in training text. Model may learn to detect the same keywords rather than clinical reasoning. |
| **Known limitations** | Published case reports use formal medical language, not patient presentation format. Pediatric override based on `age` field may be noisy. |

---

### 3. `symptom2disease`

| Property | Value |
|:---|:---|
| **Adapter** | `meditriage/builder/adapters/symptom2disease.py` |
| **Specialist label source** | `label` column (disease name) |
| **Specialist label type** | **MAPPED** |
| **Specialist mapping** | Hardcoded 19-disease mapping in adapter (e.g., "Pneumonia" → `CARDIO_PULM`, "Migraine" → `NEURO`, "Arthritis" → `ORTHO`). Unmapped diseases default to `GEN_MED`. |
| **Severity label source** | — |
| **Severity label type** | **NONE** (always `None`) |
| **Known leakage risks** | MEDIUM — Disease name in `label` column is not in `raw_text`, but symptom descriptions may strongly correlate with specific diseases. |
| **Known limitations** | Only ~1.2K rows across 19 diseases. Limited disease coverage. |

---

### 4. `chatdoctor_healthcaremagic`

| Property | Value |
|:---|:---|
| **Adapter** | `meditriage/builder/adapters/chatdoctor_healthcaremagic.py` |
| **Specialist label source** | Keyword-regex on `input (patient) + output (doctor)` text |
| **Specialist label type** | **INFERRED** |
| **Specialist mapping** | Vectorized keyword taxonomy: pediatrics → OB/GYN → neurology → cardio → ortho → GI → urology → ENT → psychiatry → oncology → surgery. Default: `GEN_MED`. |
| **Severity label source** | — |
| **Severity label type** | **NONE** (always `None`) |
| **Known leakage risks** | HIGH — Doctor `output` text (which names specialties explicitly) is used for label assignment, but only patient `input` is used as `raw_text`. The label may reflect information unavailable at inference time. |
| **Known limitations** | Conversational format. Patient questions may be vague or multi-topic. |

---

### 5. `chatdoctor_icliniq`

| Property | Value |
|:---|:---|
| **Adapter** | `meditriage/builder/adapters/chatdoctor_icliniq.py` |
| **Specialist label source** | Keyword-regex on `input + answer_icliniq/answer_chatdoctor` text |
| **Specialist label type** | **INFERRED** |
| **Specialist mapping** | Row-level keyword matching: pediatrics → OB/GYN → neurology → cardio → ortho → GI → urology → ENT → psychiatry → oncology → surgery. Default: `GEN_MED`. |
| **Severity label source** | — |
| **Severity label type** | **NONE** (always `None`) |
| **Known leakage risks** | HIGH — Same doctor-answer leakage as `chatdoctor_healthcaremagic`. |
| **Known limitations** | Only ~7.3K rows. Same conversational caveats. |

---

### 6. `neiss`

| Property | Value |
|:---|:---|
| **Adapter** | `meditriage/builder/adapters/neiss.py` |
| **Specialist label source** | `Diagnosis` code + `Body_Part` code + `Narrative_1` text regex |
| **Specialist label type** | **MAPPED** (codes) + **INFERRED** (narrative fallback) |
| **Specialist mapping** | Three-tier hierarchy: (1) Diagnosis numeric codes (e.g., 57=Fracture → `ORTHO`, 52=Concussion → `NEURO`), (2) Body_Part codes for unmapped diagnoses, (3) Narrative regex for remaining `GEN_MED` cases. Age < 18 → `PEDS` override. |
| **Severity label source** | — |
| **Severity label type** | **NONE** (always `None`) |
| **Known leakage risks** | LOW — Primary labels come from structured codes, not text. Narrative regex is only a fallback for residual cases. |
| **Known limitations** | All records are injury-related. No medical illness coverage. Text is brief ED narratives. ~70% of total dataset volume. |

---

### 7. `nhamcs_ed`

| Property | Value |
|:---|:---|
| **Adapter** | `meditriage/builder/adapters/nhamcs_ed.py` |
| **Specialist label source** | Hardcoded `"ED"` for all rows |
| **Specialist label type** | **DIRECT** (but constant — no routing signal) |
| **Specialist mapping** | All rows → `"ED"`. No specialty differentiation. |
| **Severity label source** | `IMMEDR` field from CDC fixed-width survey data |
| **Severity label type** | **DIRECT** (ESI 1-5) |
| **Severity mapping** | `IMMEDR` values "1"-"5" (or "01"-"05") → triage_level string. Invalid/missing values → `None`. |
| **Known leakage risks** | LOW — ESI scores come from independent triage assessment, not from text content. |
| **Known limitations** | Text is synthesized from coded fields: "Age: X, Sex: Y, Reason for Visit 1 (Code): ZZZZ". Not natural language. All specialist labels are `"ED"` — provides zero specialist routing signal. |

---

### 8. `fedmml_ed_triage`

| Property | Value |
|:---|:---|
| **Adapter** | `meditriage/builder/adapters/fedmml_ed_triage.py` |
| **Specialist label source** | Hardcoded `"ED"` for all rows |
| **Specialist label type** | **DIRECT** (but constant — no routing signal) |
| **Specialist mapping** | All rows → `"ED"`. No specialty differentiation. |
| **Severity label source** | `esi_level` field from dataset |
| **Severity label type** | **DIRECT** (ESI 1-5) |
| **Severity mapping** | `esi_level` → integer 1-5. Invalid values → `None`. |
| **Known leakage risks** | MEDIUM — Dataset is **SYNTHETIC** (LLM-generated). ESI labels may reflect generation artifacts rather than real clinical triage patterns. |
| **Known limitations** | Synthetic data. All specialist labels are `"ED"`. Provides 63% of all severity-labeled rows. |

---

### 9. `kaggle_medical_triage`

| Property | Value |
|:---|:---|
| **Adapter** | `meditriage/builder/adapters/kaggle_medical_triage.py` |
| **Specialist label source** | `primary_specialty` field (or default `"ED"`) |
| **Specialist label type** | **DIRECT / MAPPED** |
| **Specialist mapping** | Uses `primary_specialty` field directly if present. Falls back to `"ED"`. |
| **Severity label source** | `urgency_level` / `label` / `triage_level` field |
| **Severity label type** | **DIRECT** |
| **Severity mapping** | Reads urgency from multiple possible column names. |
| **Known leakage risks** | MEDIUM — `raw_text` includes "Clinical Reasoning" and "Recommendation" fields which may directly leak the triage decision. |
| **Known limitations** | Only ~1.1K rows. Schema varies across file formats (JSON/CSV/Parquet). Field names are not guaranteed consistent. |

---

### 10. `meddialog_en`

| Property | Value |
|:---|:---|
| **Adapter** | `meditriage/builder/adapters/meddialog_en.py` |
| **Specialist label source** | Keyword-regex on combined `instruction + input + output` text |
| **Specialist label type** | **INFERRED** |
| **Specialist mapping** | `_classify_department()` method: keyword taxonomy matching pediatric → OB/GYN → oncology → cardio → neuro → ortho → GI → urology → ENT → psychiatry → surgery. Default: `GEN_MED`. |
| **Severity label source** | — |
| **Severity label type** | **NONE** (always `None`) |
| **Known leakage risks** | HIGH — Specialist keywords used for label assignment appear in training text. Dialog format includes both patient and doctor utterances. |
| **Known limitations** | Conversational format. ~2.7M rows but many are short exchanges. Labels assigned from full dialog text but model trains on same text. |

---

## Summary Table

| Dataset | Specialist Type | Severity Type | Leakage Risk |
|:---|:---:|:---:|:---:|
| `mtsamples` | MAPPED | NONE | LOW |
| `pmc_patients` | INFERRED | NONE | HIGH |
| `symptom2disease` | MAPPED | NONE | MEDIUM |
| `chatdoctor_healthcaremagic` | INFERRED | NONE | HIGH |
| `chatdoctor_icliniq` | INFERRED | NONE | HIGH |
| `neiss` | MAPPED + INFERRED | NONE | LOW |
| `nhamcs_ed` | DIRECT (constant ED) | DIRECT | LOW |
| `fedmml_ed_triage` | DIRECT (constant ED) | DIRECT | MEDIUM |
| `kaggle_medical_triage` | DIRECT/MAPPED | DIRECT | MEDIUM |
| `meddialog_en` | INFERRED | NONE | HIGH |

### Severity Label Coverage

| Source | Rows (approx.) | % of Total |
|:---|:---:|:---:|
| `nhamcs_ed` | 50,548 | 0.5% |
| `fedmml_ed_triage` | 87,234 | 0.8% |
| `kaggle_medical_triage` | 1,112 | <0.01% |
| **Total with severity** | **~138,894** | **~1.3%** |
| **Total without severity** | **~10,395,000** | **~98.7%** |
