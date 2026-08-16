# Dataset Robustness Matrix

**Version:** v2.0.0
**Last Updated:** 2026-08-16

---

## Classification Key

| Status | Meaning |
|--------|---------|
| **SOURCE_AVAILABLE** | Genuine examples exist in approved source datasets |
| **AUGMENTATION_AVAILABLE** | Pipeline code can deterministically generate examples |
| **BOTH** | Source examples AND augmentation capability exist |
| **MISSING** | Neither source data nor augmentation capability exists |

> **Critical Rule:** The existence of generator code does NOT constitute dataset coverage. A category is SOURCE_AVAILABLE only if genuine examples exist in the approved source data.

---

## Robustness Matrix

| Category | Status | Source Evidence | Augmentation Engine | Pilot Coverage | Notes |
|----------|--------|----------------|-------------------|----------------|-------|
| **Spelling Variation** | AUGMENTATION_AVAILABLE | ❌ No dedicated source | `LexicalVariationGenerator` | ABSENT | Synonym substitutions for clinical terms |
| **Phonetic Variation** | AUGMENTATION_AVAILABLE | ❌ No dedicated source | `ColloquialIndianGenerator` | ABSENT | Indian-English phonetic patterns |
| **Roman Hindi** | AUGMENTATION_AVAILABLE | ❌ No clinical source | `OfflineMultilingualProvider` (hi-Latn) | ABSENT | 14 clinical terms × 3 templates |
| **Hinglish** | AUGMENTATION_AVAILABLE | ❌ No clinical source | `OfflineMultilingualProvider` (hi-en) | ABSENT | Code-mixed clinical templates |
| **Code Switching** | AUGMENTATION_AVAILABLE | ❌ No clinical source | `OfflineMultilingualProvider` (en-hi) | ABSENT | Clinical English with Hindi phrasing |
| **Informal Language** | AUGMENTATION_AVAILABLE | ❌ No dedicated source | `ConversationalVariationGenerator` | ABSENT | Patient-spoken style variations |
| **Clinical Shorthand** | AUGMENTATION_AVAILABLE | ❌ No dedicated source | `AbbreviatedNotationGenerator` | ABSENT | Abbreviated clinical notation |
| **ASR-like Noise** | MISSING | ❌ No source | ❌ No generator | ABSENT | No speech-to-text noise simulation |
| **Negation** | SOURCE_AVAILABLE | ✅ NEISS narratives contain natural negation | None dedicated | PRESENT (implicit) | "NO LOC" / "denies pain" in NEISS narratives |
| **Temporal Expressions** | SOURCE_AVAILABLE | ✅ NEISS/NHAMCS contain natural temporal refs | None dedicated | PRESENT (implicit) | "yesterday" / "2 hours ago" in narratives |
| **Severity Modifiers** | SOURCE_AVAILABLE | ✅ NEISS narratives contain severity language | None dedicated | PRESENT (implicit) | "severe" / "mild" / "acute" in narratives |
| **Rare Symptoms** | SOURCE_AVAILABLE | ✅ NEISS has long-tail injury narratives | `PhenotypeEngine` | PRESENT (partial) | 7.3M NEISS narratives contain rare injuries |
| **Long-tail Departments** | SOURCE_AVAILABLE | ⚠️ SURGERY (1.4%), GI (0.8%) present | None | PRESENT (sparse) | 4 departments are absent entirely |
| **Hard Negatives** | AUGMENTATION_AVAILABLE | ❌ No dedicated source | `DifferentialDiagnosisLibrary` | ABSENT | Confusion pairs for differential diagnosis |
| **Long Inputs** | SOURCE_AVAILABLE | ✅ MTSamples has lengthy transcriptions | None | PRESENT | 56 records flagged as long in pilot QC |
| **Late-occurring Red Flags** | MISSING | ❌ No labeled source | ❌ No generator | ABSENT | Requires annotated red-flag timing data |

---

## Coverage Summary

| Status | Count | Categories |
|--------|-------|-----------|
| SOURCE_AVAILABLE | 5 | Negation, temporal expressions, severity modifiers, rare symptoms, long inputs |
| AUGMENTATION_AVAILABLE | 7 | Spelling variation, phonetic variation, Roman Hindi, Hinglish, code switching, informal language, clinical shorthand, hard negatives |
| BOTH | 0 | — |
| MISSING | 2 | ASR-like noise, late-occurring red flags |

---

## Augmentation Engine Inventory

### Variation Engine (`meditriage/multilingual/variation/`)

| Generator | Robustness Category | Variants per Record |
|-----------|-------------------|-------------------|
| `LexicalVariationGenerator` | Spelling variation | Up to 2 |
| `SyntacticVariationGenerator` | Syntactic variation | Up to 2 |
| `ConversationalVariationGenerator` | Informal language | Up to 2 |
| `EdTriageVariationGenerator` | ED-specific phrasing | Up to 2 |
| `PhysicianNoteVariationGenerator` | Formal clinical style | Up to 1 |
| `NurseIntakeVariationGenerator` | Nurse intake style | Up to 1 |
| `AbbreviatedNotationGenerator` | Clinical shorthand | Up to 1 |
| `FormalDocumentationGenerator` | Formal documentation | Up to 1 |
| `ColloquialIndianGenerator` | Phonetic/colloquial Indian | Up to 2 |

**Total max variants per source record:** 8 (configurable)

### Multilingual Provider (`meditriage/multilingual/providers/offline.py`)

| Target Language | Clinical Terms | Templates | Deterministic |
|----------------|---------------|-----------|--------------|
| Hindi (Devanagari) | 14 | 3 | ✅ Yes |
| Roman Hindi | 14 | 3 | ✅ Yes |
| Hinglish (hi-en) | 14 | 3 | ✅ Yes |
| Code-Switched (en-hi) | 14 | 3 | ✅ Yes |

### Hard Negative Engine (`meditriage/multilingual/hard_negative/`)

| Component | Purpose |
|-----------|---------|
| `DifferentialDiagnosisLibrary` | Clinically plausible confusion pairs |
| `TruePhenotypeDifferentialMapping` | Phenotype-based differential diagnosis |

### Phenotype Engine (`meditriage/multilingual/phenotype/`)

| Component | Purpose |
|-----------|---------|
| `PhenotypeEngine` | Clinical phenotype variation |
| `ClinicalRules` | Semantic preservation validation |

---

## Gap Analysis

### ASR-like Noise (MISSING)

No infrastructure exists to simulate speech-to-text artifacts (e.g., homophone substitution, missing punctuation, recognition errors). This would require a dedicated noise injection module.

### Late-occurring Red Flags (MISSING)

No labeled data or annotation infrastructure exists to identify "buried" critical findings in long clinical narratives. This would require expert annotation of temporal red-flag positioning.

---

## L3Cube HingLID Status

**Role:** AUXILIARY LANGUAGE-ID RESOURCE

- 1M word-level HI/EN language identification labels
- NOT clinical triage training data
- NOT included in canonical supervised dataset
- Can inform: language ID models, code-switch detection, Hinglish pattern recognition

---

## Triagegeist Status

**Status:** RESTRICTED

- Non-Commercial Research License (Laitinen-Fredriksson Foundation)
- Redistribution: PROHIBITED
- NOT downloaded, NOT incorporated
- Contains synthetic ESI 1-5 labels; useful as reference only
