# Gate 2 — Dataset Candidate Matrix

**Specification Baseline:** MediTriageAI v1.0.0-FROZEN  
**Gate:** GATE 2 — Canonical Dataset Reconstruction  
**Date:** 2026-08-16  
**Status:** COMPLETE

---

## Scoring Methodology

Each candidate is scored 0–3 on 20 dimensions. Scores are **not** summed into a composite; they inform a qualitative decision.

| Score | Meaning |
|---|---|
| 0 | None / Not applicable |
| 1 | Low / Marginal |
| 2 | Moderate / Useful |
| 3 | High / Excellent |

---

## Candidate Scoring Matrix

### Existing Repository Datasets

| # | Dimension | MTSamples | NEISS | NHAMCS ED | Symptom2Disease | MedQA-USMLE | Med Meadow | FedMML ED | ChatDoctor HCM | ChatDoctor iCliniq | PMC-Patients | Kaggle Triage | L3Cube HingLID | MedDialog EN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Clinical relevance | 3 | 2 | 2 | 1 | 1 | 1 | 2 | 2 | 2 | 3 | 1 | 0 | 2 |
| 2 | Triage relevance | 2 | 1 | 3 | 1 | 0 | 0 | 3 | 1 | 1 | 1 | 2 | 0 | 1 |
| 3 | Label quality | 3 | 2 | 2 | 2 | 0 | 0 | 1 | 1 | 1 | 1 | 1 | 0 | 1 |
| 4 | Severity availability | 0 | 0 | 3 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 1 | 0 | 0 |
| 5 | Specialist availability | 3 | 2 | 1 | 2 | 0 | 0 | 1 | 1 | 1 | 1 | 1 | 0 | 1 |
| 6 | Language diversity | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 |
| 7 | Hinglish/code-mix value | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 |
| 8 | Linguistic diversity | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 2 | 1 | 3 | 0 |
| 9 | Provenance quality | 3 | 3 | 3 | 2 | 3 | 2 | 1 | 1 | 1 | 3 | 1 | 3 | 1 |
| 10 | Size | 1 | 3 | 1 | 1 | 1 | 1 | 2 | 2 | 1 | 2 | 1 | 1 | 3 |
| 11 | Duplication risk | 1 | 1 | 1 | 1 | 2 | 2 | 1 | 2 | 2 | 1 | 1 | 1 | 3 |
| 12 | Leakage risk | 1 | 1 | 1 | 1 | 2 | 2 | 1 | 2 | 2 | 1 | 1 | 1 | 2 |
| 13 | Legal usability | 3 | 3 | 3 | 3 | 3 | 3 | 1 | 0 | 0 | 2 | 3 | 3 | 1 |
| 14 | Reproducibility | 3 | 3 | 3 | 3 | 3 | 3 | 2 | 2 | 2 | 3 | 3 | 3 | 2 |
| 15 | Documentation quality | 2 | 2 | 3 | 1 | 3 | 2 | 1 | 1 | 1 | 3 | 1 | 2 | 1 |
| 16 | Long-tail coverage | 3 | 1 | 1 | 2 | 2 | 1 | 1 | 2 | 2 | 3 | 1 | 0 | 2 |
| 17 | Hard-negative value | 2 | 1 | 1 | 1 | 2 | 1 | 1 | 1 | 1 | 2 | 1 | 0 | 1 |
| 18 | Safety/red-flag value | 2 | 2 | 2 | 0 | 1 | 0 | 1 | 1 | 1 | 2 | 0 | 0 | 1 |
| 19 | OOD/robustness value | 1 | 1 | 1 | 1 | 2 | 1 | 1 | 1 | 1 | 1 | 1 | 2 | 0 |
| 20 | Integration cost | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 2 |

### External Candidate Datasets

| # | Dimension | MIMIC-IV-ED | MIETIC | Triagegeist |
|---|---|---|---|---|
| 1 | Clinical relevance | 3 | 3 | 2 |
| 2 | Triage relevance | 3 | 3 | 3 |
| 3 | Label quality | 3 | 3 | 2 |
| 4 | Severity availability | 3 | 3 | 3 |
| 5 | Specialist availability | 2 | 2 | 1 |
| 6 | Language diversity | 0 | 0 | 0 |
| 7 | Hinglish/code-mix value | 0 | 0 | 0 |
| 8 | Linguistic diversity | 0 | 0 | 0 |
| 9 | Provenance quality | 3 | 3 | 2 |
| 10 | Size | 3 | 2 | 2 |
| 11 | Duplication risk | 1 | 1 | 1 |
| 12 | Leakage risk | 1 | 1 | 1 |
| 13 | Legal usability | 1 | 1 | 3 |
| 14 | Reproducibility | 3 | 3 | 3 |
| 15 | Documentation quality | 3 | 3 | 2 |
| 16 | Long-tail coverage | 2 | 2 | 1 |
| 17 | Hard-negative value | 2 | 2 | 1 |
| 18 | Safety/red-flag value | 2 | 2 | 1 |
| 19 | OOD/robustness value | 2 | 2 | 1 |
| 20 | Integration cost | 1 | 1 | 2 |

---

## Existing Dataset Decisions

### KEEP

| Dataset | Rows (Historical) | Decision | Role | Rationale |
|---|---|---|---|---|
| **MTSamples** | 2,371 | **KEEP** | PRIMARY_TRAINING | CC0 license, real clinical transcriptions, genuine specialist-label mapping via `src/specialty_mapping.py`, high provenance quality. Small but high-quality anchor dataset. |
| **NEISS** | 7,137,339 | **KEEP_AFTER_TRANSFORMATION** | PRIMARY_TRAINING | Public domain, real ED injury narratives. Must be **downsampled** — 70% dominance distorts class distribution. Must fix PEDS override (age<18 overrides all departments). Adapter exists and works. |
| **NHAMCS ED** | 41,509 | **KEEP** | PRIMARY_TRAINING | CDC public-use data, real ED visits with **ESI triage levels** (1–5). One of only two sources with genuine severity labels. |
| **Symptom2Disease** | 1,153 | **KEEP** | SECONDARY_TRAINING | CC0, small symptom-disease mapping dataset. Low triage relevance but covers disease-to-department routing. |
| **Kaggle Medical Triage** | 1,112 | **KEEP_AFTER_TRANSFORMATION** | SECONDARY_TRAINING | CC0, has severity labels (non-standard string format). Must map string severity to ESI 1–5. Very small. |

### KEEP_AS_AUXILIARY

| Dataset | Rows (Historical) | Decision | Role | Rationale |
|---|---|---|---|---|
| **MedQA-USMLE** | 11,449 | **KEEP_AS_AUXILIARY** | REFERENCE_ONLY | MIT license, medical knowledge Q&A. Zero triage/severity labels. Not suitable for primary training but useful for medical vocabulary exposure. |
| **Medical Meadow MedQA** | 10,178 | **KEEP_AS_AUXILIARY** | REFERENCE_ONLY | CC-BY, Q&A format. No triage labels, no severity, no department. Useful for medical vocabulary only. |
| **L3Cube HingLID** | 37,725 | **KEEP_AS_AUXILIARY** | AUXILIARY_LANGUAGE | MIT license. Not medical content — token-level language-ID annotations for Hinglish. Valuable for Hinglish linguistic patterns but NOT as a clinical training source. Should be used for augmentation reference/linguistic patterns, not as direct training data for triage. |
| **PMC-Patients** | 167,034 | **KEEP_AS_AUXILIARY** | SECONDARY_TRAINING | CC BY-NC-SA 4.0 — **non-commercial restriction**. Real clinical case reports from PubMed Central. High clinical quality. Must comply with NC-SA terms. Research use only. |

### QUARANTINE

| Dataset | Rows (Historical) | Decision | Role | Rationale |
|---|---|---|---|---|
| **FedMML ED Triage** | 84,177 | **QUARANTINE** | QUARANTINE pending license clarification | No clear license. 100% LLM-generated synthetic data. 74% of all severity labels in historical dataset came from this source. Text provenance = Category C. **Cannot be used until license is confirmed.** If license is confirmed, may be useful as auxiliary severity-labeled source with mandatory `provenance=C` marking. |
| **MedDialog EN** | 2,616,894 | **QUARANTINE** | QUARANTINE pending CJK disposition + license | **100% of rows contain CJK characters.** 98.8% are >50% CJK (pure Chinese text mislabeled as `language=en`). License unknown. This is NOT an English medical dialogue dataset — it is a Chinese medical dialogue dataset that was mislabeled. |

### DISCARD

| Dataset | Rows (Historical) | Decision | Role | Rationale |
|---|---|---|---|---|
| **ChatDoctor HealthCareMagic** | 112,002 | **DISCARD** | REJECTED | License Grade E — scraped from commercial medical Q&A platform without verified consent. Apache 2.0 tag applied by third-party uploader, contradicted by original project's non-commercial restriction. High legal and ethical risk. |
| **ChatDoctor iCliniq** | 7,321 | **DISCARD** | REJECTED | Same legal/ethical issues as HealthCareMagic. Scraped from commercial telemedicine platform. |

---

## CJK Contamination Investigation

### Evidence

**Source dataset:** `meddialog_en`  
**CJK audit results (full 2,616,894-row scan):**

| Category | Count | Percentage |
|---|---|---|
| Total meddialog_en rows | 2,616,894 | 100% |
| Rows containing CJK characters | 2,616,894 | **100.00%** |
| Pure CJK (>50% CJK characters) | 2,586,684 | **98.85%** |
| Mixed CJK/Latin (<50% CJK) | 30,210 | **1.15%** |

### Finding

**The entire `meddialog_en` source is Chinese medical dialogue text.** It is NOT English text with occasional CJK contamination — it is a Chinese-language dataset that was ingested through an adapter that hardcoded `language="en"` without script validation.

The MedDialog project (by Zeng et al.) contains both English and Chinese medical dialogues. The adapter (`meddialog_en.py`) was designed to ingest the English portion, but the raw data downloaded to `datasets/raw/meddialog_en/` appears to contain the Chinese portion (or a mixed corpus where the Chinese portion overwhelmingly dominates).

The v2.0 adapter includes a CJK safety filter (line 85-87: rejects records with any `[\u4e00-\u9fff]` characters), but this was apparently added **after** the historical dataset was built.

### Decision

**QUARANTINE the entire `meddialog_en` source.**

Rationale:
1. **License is UNKNOWN** — cannot use regardless of content quality.
2. **100% CJK content** — this is Chinese medical text, not English.
3. **Zero triage/severity labels** — provides no supervised signal.
4. **Zero department labels of value** — departments were assigned by keyword matching against Chinese text translated to English, producing unreliable mappings.
5. **Not relevant to current scope** — the frozen specification (SPEC-01) targets English and Hinglish. Chinese is explicitly out of scope.

**The 30,210 mixed CJK/Latin rows** could theoretically contain bilingual clinical content, but given the license uncertainty and scope constraints, they should remain quarantined until:
- License is confirmed with original publisher
- A Chinese-language scope extension is approved via Change Request
- Content quality is independently validated

### Impact on Historical Dataset

Removing `meddialog_en` (2,616,894 rows = 25.58%) and `chatdoctor_*` (119,323 rows = 1.17%) from the historical 10.23M-row dataset leaves approximately **7,494,047 rows** from legitimate sources. However, the historical dataset artifact remains NOT ELIGIBLE due to the other Gate 1 failures (bypassed augmentation, missing provenance metadata, etc.).

---

## External Candidate Decisions

| Dataset | Decision | Rationale |
|---|---|---|
| **MIMIC-IV-ED** | **RESTRICTED** | Gold-standard ED dataset with ESI triage labels. Cannot be incorporated without completing PhysioNet credentialing (CITI training + institutional DUA). Recorded for future consideration. |
| **MIETIC** | **RESTRICTED** | Same access requirements as MIMIC-IV-ED. |
| **Triagegeist** | **SELECT-AS-AUXILIARY** | CC0 synthetic ED dataset with ESI labels, demographics, vitals, chief complaints. Useful as auxiliary severity-labeled source. Must be marked as Category C (synthetic). Not yet acquired — requires Kaggle download. |

---

## Final Candidate Summary

### Selected for Canonical Build (Grade A License)

| Source | Role | Rows (Est.) | Has Severity | Has Department | Language |
|---|---|---|---|---|---|
| MTSamples | PRIMARY_TRAINING | ~2,400 | No | Yes (high-quality) | en |
| NEISS (downsampled) | PRIMARY_TRAINING | TBD (cap needed) | No | Yes (inferred) | en |
| NHAMCS ED | PRIMARY_TRAINING | ~41,500 | **Yes (ESI 1–5)** | ED only | en |
| Symptom2Disease | SECONDARY_TRAINING | ~1,200 | No | Yes (mapped) | en |
| Kaggle Medical Triage | SECONDARY_TRAINING | ~1,100 | Yes (string→ESI) | Partial | en |

### Auxiliary Sources

| Source | Role | Notes |
|---|---|---|
| MedQA-USMLE | REFERENCE_ONLY | Medical vocabulary exposure only |
| Medical Meadow MedQA | REFERENCE_ONLY | Medical vocabulary exposure only |
| L3Cube HingLID | AUXILIARY_LANGUAGE | Hinglish linguistic patterns (non-medical) |
| PMC-Patients | SECONDARY_TRAINING (NC-SA) | Research use only; high clinical quality |
| Triagegeist | AUXILIARY (if acquired) | Synthetic ESI-labeled ED data |

### Quarantined

| Source | Reason |
|---|---|
| FedMML ED Triage | License unknown; 100% LLM-generated |
| MedDialog EN | License unknown; 100% CJK content mislabeled as English |

### Rejected

| Source | Reason |
|---|---|
| ChatDoctor HealthCareMagic | License Grade E — scraped commercial data |
| ChatDoctor iCliniq | License Grade E — scraped commercial data |
