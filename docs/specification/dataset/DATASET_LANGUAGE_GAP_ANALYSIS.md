# Dataset Language Gap Analysis

**Version:** v2.0.0
**Last Updated:** 2026-08-16
**Pilot Reference:** `meditriage/data/canonical/pilot/dataset.parquet` (1,398 rows)

---

## Summary

The current pilot dataset is **100% English**. The frozen v1.0.0 specification requires multilingual support (English, Hindi, Hinglish/code-mixed). This document audits the gap between current state and specification requirements.

---

## Language Coverage Matrix

| Language | Code | Source Available | Source Count | Augmentation Available | In Pilot | Training Eligible | Notes |
|----------|------|-----------------|-------------|----------------------|----------|-------------------|-------|
| English | `en` | ✅ YES | 1,398 (pilot) / millions (full) | N/A | ✅ YES | ✅ YES | All 5 approved sources |
| Hindi (Devanagari) | `hi` | ❌ NO | 0 | ✅ YES | ❌ NO | Via augmentation only | Offline provider: 14 clinical terms × 3 templates |
| Roman Hindi | `hi-Latn` | ❌ NO | 0 | ✅ YES | ❌ NO | Via augmentation only | Offline provider: deterministic transliteration |
| Hinglish (mixed) | `hi-en` | ❌ NO | 0 | ✅ YES | ❌ NO | Via augmentation only | Offline provider: code-mixed templates |
| Code-Switched (En-Hi) | `en-hi` | ❌ NO | 0 | ✅ YES | ❌ NO | Via augmentation only | Offline provider: clinical phrasing templates |
| CJK | — | ❌ QUARANTINED | 0 | ❌ NO | ❌ NO | ❌ NO | MedDialog_EN was 100% CJK; quarantined |
| Other | — | ❌ NO | 0 | ❌ NO | ❌ NO | ❌ NO | No other languages in approved sources |

---

## Detailed Assessment

### English — SOURCE_AVAILABLE

- **Sources:** NEISS (7.3M), NHAMCS-ED (50K), MTSamples (~5K), Symptom2Disease (1.2K), Kaggle Medical Triage (4K)
- **Status:** Abundant. All pilot and full-build sources are English.
- **Clinical relevance:** High. NEISS/NHAMCS are genuine clinical ED data.
- **Quality:** Native English; no translation artifacts.

### Hindi (Devanagari) — AUGMENTATION_AVAILABLE

- **Source data:** None. No approved dataset contains native Hindi clinical text.
- **Augmentation:** `meditriage/multilingual/providers/offline.py` provides deterministic rule-based generation using:
  - 14 clinical symptom terms with Devanagari translations
  - 3 sentence templates per language variant
  - Duration expressions in Devanagari numerals
- **Clinical relevance:** Template-generated; covers common ED complaints but lacks diversity of genuine clinical notes.
- **Limitation:** Augmented Hindi is synthetic, not source data.

### Roman Hindi — AUGMENTATION_AVAILABLE

- **Source data:** None.
- **Augmentation:** Same offline provider with Latin-script Hindi transliterations.
- **Clinical relevance:** Covers common transliterated medical terms.
- **Limitation:** Template-based; limited vocabulary breadth.

### Hinglish / Code-Mixed — AUGMENTATION_AVAILABLE

- **Source data:** None genuinely clinical.
- **L3Cube HingLID:** Available at `datasets/raw/l3cube_code_mixed/` (1M token-level HI/EN labels). However, this is a **word-level language identification dataset**, NOT sentence-level clinical text. Its role is AUXILIARY LANGUAGE-ID RESOURCE only.
- **Augmentation:** Offline provider generates code-mixed clinical templates (e.g., "Patient ko chest mein pain ho raha hai").
- **Clinical relevance:** Limited. Generated code-mixing follows fixed patterns.

### CJK — QUARANTINED

- **MedDialog_EN:** Gate 1 forensic audit established this dataset is 100% CJK despite the "EN" designation. Permanently quarantined.
- **No CJK support is planned** in the v1.0.0 specification.

---

## Existing Multilingual Infrastructure

| Component | Location | Status | Capabilities |
|-----------|----------|--------|-------------|
| Offline Provider | `meditriage/multilingual/providers/offline.py` | IMPLEMENTED | 4 target languages, 14 clinical terms, deterministic |
| Gemini Provider | `meditriage/multilingual/providers/` | STUB | Requires API key; not used in pilot |
| Variation Engine | `meditriage/multilingual/variation/` | IMPLEMENTED | 9 variation styles (lexical, syntactic, conversational, ED triage, physician note, nurse intake, abbreviated, formal, colloquial Indian) |
| Phenotype Engine | `meditriage/multilingual/phenotype/` | IMPLEMENTED | Differential diagnosis pairs, clinical rules |
| Hard Negative Engine | `meditriage/multilingual/hard_negative/` | IMPLEMENTED | Differential diagnosis confusion pairs |
| Translator | `meditriage/multilingual/translator.py` | IMPLEMENTED | Orchestrates translation with quality validation |
| Validator | `meditriage/multilingual/validator.py` | IMPLEMENTED | Translation quality scoring |

---

## External Multilingual Dataset Candidates

### Clinical Multilingual Sources

| Dataset | Language | License | Clinical | Status | Notes |
|---------|----------|---------|----------|--------|-------|
| IIIT-H Hindi Health QA | Hindi | Unknown | Partial | REQUIRES INVESTIGATION | Hindi medical QA; access/license unverified |
| IIT Patna Hindi Medical NER | Hindi | CC-BY-4.0 | Yes | CANDIDATE | Named entity recognition; limited text |
| Hindi Medical Translation (OPUS) | Hindi-English | Various | Partial | AUXILIARY | Parallel medical text; not triage-specific |

### General Multilingual Sources (Non-Clinical)

| Dataset | Language | License | Clinical | Status | Notes |
|---------|----------|---------|----------|--------|-------|
| L3Cube HingLID | Hinglish | CC-BY-4.0 | ❌ No | AUXILIARY LANGUAGE-ID | Token-level HI/EN labels; not clinical |
| LINCE Hindi-English | Code-mixed | CC-BY-4.0 | ❌ No | AUXILIARY | Social media code-switching; not medical |
| GLUECoS | Code-mixed | Research | ❌ No | RESTRICTED | NLU benchmark; not clinical |

---

## Critical Distinction

> **AUGMENTATION CAPABILITY ≠ SOURCE COVERAGE**
>
> The existence of code that can generate multilingual text does NOT constitute evidence that the dataset contains genuine multilingual clinical data. All multilingual content in the current pipeline would be **augmented/synthetic**, not **source**.
>
> The final dataset documentation must clearly label:
> - `provenance: SOURCE` for genuine multilingual data
> - `provenance: AUGMENTED` for generated multilingual data

---

## Recommendations

1. **Do not fabricate** multilingual source data claims.
2. **Do use** the existing augmentation infrastructure to generate controlled multilingual variants.
3. **Mark** all augmented multilingual records with proper provenance tracking.
4. **Investigate** IIT Patna Hindi Medical NER and IIIT-H datasets for genuine Hindi clinical text.
5. **Maintain** L3Cube HingLID as an auxiliary language-ID training resource, not as clinical triage data.
6. **Accept** that the v1.0.0 pilot is English-only; multilingual expansion is a planned augmentation stage.
