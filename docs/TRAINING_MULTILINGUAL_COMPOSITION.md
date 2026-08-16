# MediTriageAI — Multilingual Training Composition and Split Balance Audit

**Specification Baseline:** `v1.0.0-FROZEN`  
**Audit Date:** `2026-08-16`  
**Dataset:** `meditriage/data/canonical/v1.0.0/dataset.parquet` (53,067 records)

---

## 1. Split Distribution by Language Code

| Language Tag | Linguistic Scope | Overall (53,067) | Train (42,414) | Validation (5,298) | Test (5,355) | Split Proportions (Train/Val/Test) |
|---|---|---|---|---|---|---|
| `en` | Standard English | 52,080 (98.1%) | 41,631 (98.2%) | 5,217 (98.5%) | 5,232 (97.7%) | 79.9% / 10.0% / 10.1% |
| `hi-en` | Hinglish Code-Mixed | 329 (0.6%) | 261 (0.6%) | 27 (0.5%) | 41 (0.8%) | 79.3% / 8.2% / 12.5% |
| `hi` | Devanagari Hindi | 329 (0.6%) | 261 (0.6%) | 27 (0.5%) | 41 (0.8%) | 79.3% / 8.2% / 12.5% |
| `hi-Latn` | Romanized Hindi | 329 (0.6%) | 261 (0.6%) | 27 (0.5%) | 41 (0.8%) | 79.3% / 8.2% / 12.5% |
| **TOTAL** | All Languages | **53,067** | **42,414** | **5,298** | **5,355** | **79.9% / 10.0% / 10.1%** |

---

## 2. Provenance & Augmentation Distribution by Split

| Provenance Category | Augmentation Type / Stratum | Overall (53,067) | Train (42,414) | Validation (5,298) | Test (5,355) |
|---|---|---|---|---|---|
| `SOURCE` (Grade-A) | Unaugmented Native Source | **40,681** | **32,544** | **4,066** | **4,071** |
| `A` (Deterministic) | `abbreviated_notation` | 3,878 | 3,106 | 378 | 394 |
| `A` (Deterministic) | `lexical_variation` | 2,149 | 1,711 | 213 | 225 |
| `A` (Deterministic) | `late_red_flag` | 1,962 | 1,582 | 188 | 192 |
| `A` (Deterministic) | `asr_noise` | 1,430 | 1,135 | 151 | 144 |
| `A` (Deterministic) | `colloquial_indian` | 1,354 | 1,058 | 156 | 140 |
| `A` (Deterministic) | `informal_variation` | 616 | 485 | 63 | 68 |
| `A` (Deterministic) | `multilingual_hinglish` | 329 | 261 | 27 | 41 |
| `A` (Deterministic) | `multilingual_hindi_devanagari` | 329 | 261 | 27 | 41 |
| `A` (Deterministic) | `multilingual_roman_hindi` | 329 | 261 | 27 | 41 |
| `A` (Deterministic) | `hard_negative` | 10 | 9 | 0 | 1 |
| **TOTAL AUGMENTED** | All Strata | **12,386** | **9,870** | **1,232** | **1,284** |

---

## 3. Split Isolation & Parent-Child Lineage Verification

1. **Split Inheritance:**
   - 100% of augmented records inherit the exact split assigned to their source parent record (`augmentation_parent_id`).
   - Split mismatches between parent and child: **0**.
2. **Cross-Split Leakage:**
   - Cross-split `source_record_id` sharing: **0**.
   - Cross-split exact normalized text duplicate matches: **0**.
3. **Distribution Uniformity:**
   - Multilingual and robustness strata are divided across train (79.7%), validation (9.9%), and test (10.4%) splits.
