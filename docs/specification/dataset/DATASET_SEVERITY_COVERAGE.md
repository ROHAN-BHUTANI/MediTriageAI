# Dataset Severity Coverage

**Version:** v2.0.0
**Last Updated:** 2026-08-16
**Source:** Empirical analysis of approved datasets

---

## Governance Rule

> **ONLY authoritative/native severity labels may populate `triage_level`.**
>
> Kaggle urgency labels (Routine/Urgent/Emergency/Observation) are stored as `quality_flags` metadata. They are NOT mapped to S1-S5.

---

## Full-Scale Severity Coverage (NHAMCS ED 2019-2021)

NHAMCS is the **only** approved source with native ESI (Emergency Severity Index) triage labels.

| Level | Count | Percentage | Source | Label Provenance | Native/Mapped |
|-------|-------|-----------|--------|-----------------|--------------|
| S1 | 527 | 1.0% | NHAMCS ED | IMMEDR field | NATIVE |
| S2 | 5,019 | 9.9% | NHAMCS ED | IMMEDR field | NATIVE |
| S3 | 17,274 | 34.2% | NHAMCS ED | IMMEDR field | NATIVE |
| S4 | 9,510 | 18.8% | NHAMCS ED | IMMEDR field | NATIVE |
| S5 | 1,384 | 2.7% | NHAMCS ED | IMMEDR field | NATIVE |
| UNKNOWN | 16,834 | 33.3% | NHAMCS ED | IMMEDR missing/invalid | N/A |
| **Total NHAMCS** | **50,548** | **100%** | | | |

### ESI Distribution (Labeled Records Only)

| Level | Count | % of Labeled | Clinical Meaning |
|-------|-------|-------------|-----------------|
| S1 | 527 | 1.6% | Resuscitation |
| S2 | 5,019 | 14.9% | Emergent |
| S3 | 17,274 | 51.3% | Urgent |
| S4 | 9,510 | 28.2% | Less Urgent |
| S5 | 1,384 | 4.1% | Non-Urgent |
| **Total Labeled** | **33,714** | **100%** | |

---

## Pilot Severity Coverage (1,398 rows)

| Level | Count | Source | Provenance |
|-------|-------|--------|-----------|
| S1 | 1 | NHAMCS ED pilot | native_esi |
| S2 | 10 | NHAMCS ED pilot | native_esi |
| S3 | 79 | NHAMCS ED pilot | native_esi |
| S4 | 9 | NHAMCS ED pilot | native_esi |
| S5 | 0 | — | — |
| UNKNOWN | 1,299 | All other sources | none |

---

## Per-Source Severity Status

| Source | Has Native Severity | Severity Type | Notes |
|--------|-------------------|---------------|-------|
| NHAMCS ED | ✅ YES | ESI 1-5 (native) | Only source with authoritative ESI labels |
| NEISS | ❌ NO | None | Injury narratives; no triage level assigned |
| MTSamples | ❌ NO | None | Transcription samples; no severity |
| Symptom2Disease | ❌ NO | None | Symptom-disease mapping; no severity |
| Kaggle Medical Triage | ⚠️ PARTIAL | Urgency labels (NOT ESI) | `Routine`/`Urgent`/`Emergency`/`Observation` stored as `quality_flags` only |

---

## Kaggle Urgency Labels (NOT ESI)

The Kaggle Medical Triage dataset contains urgency labels that are explicitly **NOT** Emergency Severity Index levels:

| Kaggle Label | Stored In | Mapped to triage_level? |
|---|---|---|
| Routine | `quality_flags` | ❌ NO |
| Urgent | `quality_flags` | ❌ NO |
| Emergency | `quality_flags` | ❌ NO |
| Observation | `quality_flags` | ❌ NO |

**Rationale:** No authoritative documentation establishes equivalence between these urgency labels and ESI 1-5. Mapping without evidence would introduce label noise.

---

## Severity Imbalance Analysis

ESI severity is naturally imbalanced:
- **S1 (Resuscitation):** Very rare (1.6% of labeled) — expected in real ED data
- **S3 (Urgent):** Most common (51.3% of labeled) — expected
- **S5 (Non-Urgent):** Uncommon (4.1% of labeled) — many non-urgent patients don't visit ED

This is the **natural clinical distribution** and should NOT be artificially balanced.

---

## Recommendations

1. **Do not manufacture** severity labels to improve balance.
2. **Do not map** Kaggle urgency labels to ESI without authoritative evidence.
3. **Accept** the natural ESI imbalance as clinically representative.
4. **Use** stratified sampling in evaluation to ensure all severity levels are tested.
5. **Consider** Triagegeist as a reference (RESTRICTED license; see note below), but do NOT incorporate it.

### Triagegeist Note

Triagegeist contains synthetic ESI 1-5 labels calibrated from MIMIC-IV-ED and NHAMCS literature. However:
- License: **Non-Commercial Research License (Laitinen-Fredriksson Foundation)**
- Redistribution: **PROHIBITED**
- Status: **RESTRICTED** — cannot be downloaded or incorporated
