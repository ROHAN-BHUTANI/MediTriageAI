# MediTriageAI Builder Production Forensic Audit Report

**Total Exported Dataset Rows**: `10,230,264`  

**Global Supervision Coverage**: `99.89%`  

**Consistency Verification Status**: `PASSED [OK]`  


## 1. Per-Adapter Ingestion & Retention Breakdown

| Dataset Name | Raw Read | Emitted | Validated | Exported | Val Loss | Dedup Loss | Retention % | Dup % | Avg Text Len | Median Text Len |
|--------------|----------|---------|-----------|----------|----------|------------|-------------|-------|--------------|-----------------|
| `mtsamples` | 4,999 | 4,999 | 4,999 | **2,371** | 0 | 2,628 | 47.43% | 52.57% | 2988.2 | 2591.0 |
| `pmc_patients` | 167,034 | 167,034 | 167,034 | **167,034** | 0 | 0 | 100.0% | 0.0% | 2765.3 | 2491.0 |
| `medqa_usmle` | 11,451 | 11,451 | 11,451 | **11,449** | 0 | 2 | 99.98% | 0.02% | 726.3 | 696.0 |
| `medical_meadow_medqa` | 10,178 | 10,178 | 10,178 | **10,178** | 0 | 0 | 100.0% | 0.0% | 909.3 | 881.0 |
| `symptom2disease` | 1,200 | 1,200 | 1,200 | **1,153** | 0 | 47 | 96.08% | 3.92% | 171.1 | 169.0 |
| `chatdoctor_healthcaremagic` | 112,156 | 112,156 | 112,156 | **112,002** | 0 | 154 | 99.86% | 0.14% | 421.1 | 352.0 |
| `chatdoctor_icliniq` | 7,321 | 7,321 | 7,321 | **7,321** | 0 | 0 | 100.0% | 0.0% | 453.1 | 347.0 |
| `neiss` | 7,326,429 | 7,326,429 | 7,315,729 | **7,137,339** | 10,700 | 178,390 | 97.42% | 2.44% | 87.6 | 78.0 |
| `nhamcs_ed` | 50,548 | 50,548 | 50,548 | **41,509** | 0 | 9,039 | 82.12% | 17.88% | 94.6 | 88.0 |
| `fedmml_ed_triage` | 87,234 | 87,234 | 87,234 | **84,177** | 0 | 3,057 | 96.5% | 3.5% | 255.5 | 251.0 |
| `kaggle_medical_triage` | 1,112 | 1,112 | 1,112 | **1,112** | 0 | 0 | 100.0% | 0.0% | 656.2 | 653.0 |
| `l3cube_code_mixed` | 38,176 | 38,176 | 38,176 | **37,725** | 0 | 451 | 98.82% | 1.18% | 151.0 | 138.0 |
| `meddialog_en` | 2,725,990 | 2,725,990 | 2,725,990 | **2,616,894** | 0 | 109,096 | 96.0% | 4.0% | 202.0 | 138.0 |

## 2. Department & Specialty Supervision Distribution

| Department / Specialty | Row Count | Percentage |
|-----------------------|-----------|------------|
| `GEN_MED` | 3,159,934 | 30.89% |
| `PEDS` | 2,691,856 | 26.31% |
| `ORTHO` | 1,813,489 | 17.73% |
| `ENT_OPHTHALMO` | 1,213,464 | 11.86% |
| `NEURO` | 638,147 | 6.24% |
| `CARDIO_PULM` | 342,508 | 3.35% |
| `ED` | 125,703 | 1.23% |
| `RENAL_URO` | 66,650 | 0.65% |
| `SURGERY` | 62,332 | 0.61% |
| `ONCOLOGY_HEME` | 56,709 | 0.55% |
| `GI` | 27,491 | 0.27% |
| `OBGYN` | 17,765 | 0.17% |
| `None` | 10,178 | 0.10% |
| `PSYCH` | 2,926 | 0.03% |
| `Cardiology` | 180 | 0.00% |
| `Emergency Medicine` | 150 | 0.00% |
| `Neurology` | 150 | 0.00% |
| `Dermatology` | 120 | 0.00% |
| `Orthopedics` | 120 | 0.00% |
| `Gastroenterology` | 100 | 0.00% |
| `Pulmonology` | 100 | 0.00% |
| `Mental Health` | 100 | 0.00% |
| `Urology` | 80 | 0.00% |
| `Endocrinology` | 12 | 0.00% |

## 3. Language Distribution

| Language Code | Row Count | Percentage |
|---------------|-----------|------------|
| `en` | 10,192,539 | 99.63% |
| `hi-en` | 37,725 | 0.37% |

## 4. Largest Data Loss Sources

### Largest Deduplication Losses (Stage 4)

- `neiss`: **178,390** duplicate rows removed
- `meddialog_en`: **109,096** duplicate rows removed
- `nhamcs_ed`: **9,039** duplicate rows removed
- `fedmml_ed_triage`: **3,057** duplicate rows removed
- `mtsamples`: **2,628** duplicate rows removed

### Largest Validation Losses (Stage 3)

- `neiss`: **10,700** rows filtered due to missing text/supervision
- `mtsamples`: **0** rows filtered due to missing text/supervision
- `pmc_patients`: **0** rows filtered due to missing text/supervision
- `medqa_usmle`: **0** rows filtered due to missing text/supervision
- `medical_meadow_medqa`: **0** rows filtered due to missing text/supervision

## 5. Pipeline Consistency Verification Checks

- `sum(adapter exports) == final dataset rows`: **PASSED**
- `sum(department counts) == dataset rows`: **PASSED**
- `sum(split counts) == dataset rows`: **PASSED**
- `no duplicate IDs`: **PASSED**
- `no duplicate raw_text after export`: **PASSED**
