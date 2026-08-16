# MediTriageAI — Training Label and Ontology Contract

**Specification Baseline:** `v1.0.0-FROZEN`  
**Document Status:** AUTHORITATIVE LABEL MAPPING CONTRACT  
**Date:** `2026-08-16`

---

## 1. Specialist Department Routing Contract

The dataset `department` field maps deterministically to model output logits via the following integer index mapping:

```
dataset.department → LabelValidator.validate_specialist() → model specialist logit index
```

| Integer Index | Department Key | Semantic Scope | Primary Sources |
|---|---|---|---|
| **0** | `CARDIO_PULM` | Cardiology, Pulmonology, Respiratory | MTSamples, Symptom2Disease |
| **1** | `ED` | Emergency Medicine, Acute Triage | NHAMCS ED, NEISS, Kaggle |
| **2** | `ENT_OPHTHALMO` | Otolaryngology, Ophthalmology | MTSamples, Symptom2Disease |
| **3** | `GEN_MED` | General Internal Medicine, Primary Care, Catch-all | MTSamples, Symptom2Disease |
| **4** | `GI` | Gastroenterology, Hepatology | MTSamples, Symptom2Disease |
| **5** | `NEURO` | Neurology, Neurosurgery | MTSamples, Symptom2Disease |
| **6** | `OBGYN` | Obstetrics & Gynecology, Women's Health | MTSamples |
| **7** | `ONCOLOGY_HEME` | Medical Oncology, Hematology | MTSamples |
| **8** | `ORTHO` | Orthopedics, Musculoskeletal, Sports Medicine | MTSamples, NEISS |
| **9** | `PEDS` | Pediatrics | MTSamples |
| **10**| `PSYCH` | Psychiatry, Behavioral Health | MTSamples, Symptom2Disease |
| **11**| `RENAL_URO` | Nephrology, Urology | MTSamples, Symptom2Disease |
| **12**| `SURGERY` | General Surgery, Post-operative Care | MTSamples |

*Specialist Unlabeled / Missing Fallback:* Encoded as `-1` (`ignore_index`).

---

## 2. ESI Severity Acuity Contract

The dataset `triage_level` field maps deterministically to model output logits via the following integer index mapping:

```
dataset.triage_level → LabelValidator.validate_severity() → model severity logit index
```

| Integer Index | Severity Key | ESI Acuity Level | Clinical Meaning | Native Sources |
|---|---|---|---|---|
| **0** | `S1` | ESI Level 1 | Resuscitation (Immediate life threat) | NHAMCS ED |
| **1** | `S2` | ESI Level 2 | Emergent (High risk, acute distress) | NHAMCS ED |
| **2** | `S3` | ESI Level 3 | Urgent (Stable, 2+ resources needed) | NHAMCS ED |
| **3** | `S4` | ESI Level 4 | Less Urgent (Stable, 1 resource needed) | NHAMCS ED |
| **4** | `S5` | ESI Level 5 | Non-Urgent (Stable, 0 resources needed) | NHAMCS ED |
| **-1** | `NULL / UNKNOWN` | **IGNORE** | Unlabeled (Masked by Focal Loss) | NEISS, MTSamples, Kaggle, S2D |

---

## 3. Machine-Readable Schema Encoding JSON

```json
{
  "version": "1.0.0",
  "specialist_ontology": {
    "num_classes": 13,
    "mapping": {
      "CARDIO_PULM": 0,
      "ED": 1,
      "ENT_OPHTHALMO": 2,
      "GEN_MED": 3,
      "GI": 4,
      "NEURO": 5,
      "OBGYN": 6,
      "ONCOLOGY_HEME": 7,
      "ORTHO": 8,
      "PEDS": 9,
      "PSYCH": 10,
      "RENAL_URO": 11,
      "SURGERY": 12
    },
    "ignore_index": -1
  },
  "severity_ontology": {
    "num_classes": 5,
    "mapping": {
      "S1": 0,
      "S2": 1,
      "S3": 2,
      "S4": 3,
      "S5": 4
    },
    "ignore_index": -1
  }
}
```
