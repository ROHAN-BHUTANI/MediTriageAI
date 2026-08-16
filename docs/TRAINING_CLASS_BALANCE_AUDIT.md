# MediTriageAI — Training Set Class Balance and Imbalance Audit

**Specification Baseline:** `v1.0.0-FROZEN`  
**Audit Date:** `2026-08-16`  
**Inspected Split:** `train` split (42,414 records in `meditriage/data/canonical/v1.0.0/dataset.parquet`)

---

## 1. Specialist Department Class Distribution (Train Split)

Total training records: **42,414**.

| Integer Index | Department Name | Train Count | % of Train Set | Imbalance Ratio (vs Majority) | Acuity / Rarity Category |
|---|---|---|---|---|---|
| **0** | `CARDIO_PULM` | 1,522 | 3.6% | 1 : 14.9 | Moderate |
| **1** | `ED` | 22,683 | 53.5% | 1 : 1.0 | **MAJORITY (53.5%)** |
| **2** | `ENT_OPHTHALMO` | 1,284 | 3.0% | 1 : 17.7 | Moderate |
| **3** | `GEN_MED` | 3,945 | 9.3% | 1 : 5.7 | Common |
| **4** | `GI` | 585 | 1.4% | 1 : 38.8 | Low |
| **5** | `NEURO` | 2,883 | 6.8% | 1 : 7.9 | Common |
| **6** | `OBGYN` | 80 | 0.2% | 1 : 283.5 | **EXTREME RARE (<0.2%)** |
| **7** | `ONCOLOGY_HEME` | 178 | 0.4% | 1 : 127.4 | **EXTREME RARE (<0.5%)** |
| **8** | `ORTHO` | 5,828 | 13.7% | 1 : 3.9 | Common |
| **9** | `PEDS` | 150 | 0.4% | 1 : 151.2 | **EXTREME RARE (<0.5%)** |
| **10**| `PSYCH` | 312 | 0.7% | 1 : 72.7 | **RARE (<1.0%)** |
| **11**| `RENAL_URO` | 774 | 1.8% | 1 : 29.3 | Low |
| **12**| `SURGERY` | 2,190 | 5.2% | 1 : 10.4 | Moderate |

---

## 2. ESI Triage Severity Class Distribution (Train Split)

Total labeled ESI records: **16,071** (37.9% of train).  
Total unlabeled / masked records: **26,343** (62.1% of train).

| Integer Index | Severity Label | Train Count | % of Total Train | % of Labeled ESI | Imbalance Ratio (vs S3 Majority) | Clinical Risk Note |
|---|---|---|---|---|---|---|
| **0** | `S1` (Resuscitation) | 261 | 0.6% | 1.6% | 1 : 32.8 | **HIGH-RISK CRITICAL RARE** |
| **1** | `S2` (Emergent) | 2,694 | 6.4% | 16.8% | 1 : 3.2 | Moderate |
| **2** | `S3` (Urgent) | 8,562 | 20.2% | 53.3% | 1 : 1.0 | **MAJORITY LABELED** |
| **3** | `S4` (Less Urgent) | 3,938 | 9.3% | 24.5% | 1 : 2.2 | Common |
| **4** | `S5` (Non-Urgent) | 616 | 1.5% | 3.8% | 1 : 13.9 | **RARE NON-URGENT** |
| **-1**| `UNKNOWN (Masked)` | 26,343 | 62.1% | — | — | Masked via `ignore_index=-1` |

---

## 3. High-Risk Imbalance Findings & Mitigations

1. **Four Extreme Rare Departments:**
   - `OBGYN` (80 records), `PEDS` (150 records), `ONCOLOGY_HEME` (178 records), `PSYCH` (312 records).
   - *Mitigation:* Multi-task Focal Loss with $\gamma = 2.0$ dynamically scales gradients $(1 - p_t)^2$ up for hard/rare classes. Class-weighted Focal Loss can also be supplied via `specialist_class_weights`.
2. **Critical Severity Extremes ($S1$ and $S5$):**
   - $S1$ represents immediate life threat (1.6% of labeled data). Misclassifying $S1$ as $S4/S5$ is clinically hazardous.
   - *Mitigation:* Evaluated using cost-sensitive Macro-F1 and directional ordinal error penalty in `scripts/evaluate.py`.
