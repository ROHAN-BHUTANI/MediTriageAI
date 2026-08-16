# Dataset Department Coverage

**Version:** v2.0.0
**Last Updated:** 2026-08-16
**Canonical Departments:** 13

---

## Pilot Coverage (1,398 rows)

| Department | Count | % | Sources | Label Provenance | Confidence | Status |
|-----------|-------|---|---------|-----------------|------------|--------|
| ORTHO | 387 | 27.7% | NEISS | Inferred (diagnosis/body_part codes + narrative regex) | Low-Medium | ✅ ADEQUATE |
| ENT_OPHTHALMO | 379 | 27.1% | NEISS | Inferred (diagnosis/body_part codes + narrative regex) | Low-Medium | ✅ ADEQUATE |
| CARDIO_PULM | 208 | 14.9% | NEISS, Kaggle | NEISS: inferred; Kaggle: mapped from specialty | Medium-High | ✅ ADEQUATE |
| GEN_MED | 124 | 8.9% | NEISS, Symptom2Disease | NEISS: fallback; S2D: mapped from disease | Low | ✅ ADEQUATE |
| ED | 99 | 7.1% | NHAMCS ED | Native (all NHAMCS records are ED) | High | ✅ ADEQUATE |
| RENAL_URO | 91 | 6.5% | NEISS | Inferred (body_part codes) | Low | ⚠️ MARGINAL |
| NEURO | 80 | 5.7% | NEISS | Inferred (diagnosis codes + narrative regex) | Low-Medium | ⚠️ MARGINAL |
| SURGERY | 19 | 1.4% | NEISS | Inferred (diagnosis codes) | Low | ⚠️ UNDERREPRESENTED |
| GI | 11 | 0.8% | NEISS | Inferred (diagnosis codes) | Low | ⚠️ UNDERREPRESENTED |
| PEDS | 0 | 0.0% | — | — | — | ❌ ABSENT |
| OBGYN | 0 | 0.0% | — | — | — | ❌ ABSENT |
| PSYCH | 0 | 0.0% | — | — | — | ❌ ABSENT |
| ONCOLOGY_HEME | 0 | 0.0% | — | — | — | ❌ ABSENT |

---

## Coverage Summary

| Status | Count | Departments |
|--------|-------|------------|
| ✅ ADEQUATE (>5%) | 5 | ORTHO, ENT_OPHTHALMO, CARDIO_PULM, GEN_MED, ED |
| ⚠️ MARGINAL (1-5%) | 2 | RENAL_URO, NEURO |
| ⚠️ UNDERREPRESENTED (<1.5%) | 2 | SURGERY, GI |
| ❌ ABSENT (0%) | 4 | PEDS, OBGYN, PSYCH, ONCOLOGY_HEME |

---

## Per-Source Department Contribution

### NEISS (999 pilot rows)

Primary contributor to most departments via diagnosis code, body part code, and narrative heuristics.

| Department | Count | Mapping Source |
|-----------|-------|---------------|
| ORTHO | 387 | Diagnosis codes (55, 57, 64) + body part + narrative |
| ENT_OPHTHALMO | 328 | Diagnosis codes (54, 58, 59) + body part (76, 77) + narrative |
| GEN_MED | 122 | Fallback (no specific mapping matched) |
| NEURO | 79 | Diagnosis codes (52, 61) + body part (75) + narrative |
| CARDIO_PULM | 52 | Diagnosis codes (65, 67, 68) + body part (31) + narrative |
| SURGERY | 19 | Diagnosis codes (50, 63) |
| RENAL_URO | 11 | Body part codes (33, 38) |
| GI | 1 | Diagnosis code (66) |

### NHAMCS ED (99 pilot rows)
- All records: **ED** (native; all NHAMCS records are ED visits)

### MTSamples (100 pilot rows)
- Mapped from `medical_specialty` field via `KAGGLE_SPECIALTY_TO_DEPT` lookup
- Covers: CARDIO_PULM, ORTHO, GEN_MED, GI, NEURO, ENT_OPHTHALMO, SURGERY, RENAL_URO

### Symptom2Disease (100 pilot rows)
- Mapped from disease-to-department heuristic
- Covers: GEN_MED, NEURO, ENT_OPHTHALMO, GI, CARDIO_PULM, RENAL_URO

### Kaggle Medical Triage (100 pilot rows)
- Mapped from `medical_specialty` field
- Pilot sample: 100% CARDIO_PULM (first 100 rows are cardiology)
- Full dataset covers: CARDIO_PULM, ED, NEURO, ENT_OPHTHALMO, ORTHO, GI, PSYCH, GEN_MED

---

## Absent Departments — Analysis

### PEDS (Pediatrics)
- **Available via:** NEISS age-based override (currently DISABLED by default)
- **If enabled:** ~26% of NEISS records would become PEDS (age < 18)
- **Decision:** PEDS override is a configurable flag (`--peds-override`), disabled pending explicit authorization
- **Improvement path:** Enable PEDS override OR source pediatric-specific ED datasets

### OBGYN (Obstetrics/Gynecology)
- **No source available** in current approved datasets
- **NEISS:** Injury-focused; very few OB/GYN relevant narratives
- **Improvement path:** Would require a dedicated OB/GYN clinical dataset

### PSYCH (Psychiatry)
- **Partially available** in Kaggle Medical Triage (full dataset has psychiatry specialty records)
- **Not in pilot** because first 100 rows are cardiology
- **Improvement path:** Full build will include Kaggle psychiatry records

### ONCOLOGY_HEME (Oncology/Hematology)
- **No source available** in current approved datasets
- **Improvement path:** Would require oncology-specific clinical data

---

## Recommendations

1. **Do not** solve department imbalance by arbitrary relabeling.
2. **Enable PEDS override** as a conscious decision (not by default) after reviewing the clinical appropriateness.
3. **The full canonical build** will naturally improve coverage for PSYCH (via full Kaggle dataset) and potentially GI, SURGERY (via full NEISS/MTSamples).
4. **Accept** that OBGYN and ONCOLOGY_HEME may remain absent unless new datasets are sourced.
5. **Label provenance** must be documented for every department assignment (native vs. inferred vs. mapped).
