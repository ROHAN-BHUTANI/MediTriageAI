# MediTriageAI Builder Production Forensic Audit Report

**Total Exported Dataset Rows**: `7,621,350`  

**Global Supervision Coverage**: `74.51%`  

**Consistency Verification Status**: `VERIFIED`  


## 1. Per-Adapter Ingestion & Retention Breakdown

| Dataset Name | Raw Read | Emitted | Validated | Exported | Val Loss | Dedup Loss | Retention % | Dup % | Avg Text Len | Median Text Len |
|--------------|----------|---------|-----------|----------|----------|------------|-------------|-------|--------------|-----------------|
| `mtsamples` | 4,999 | 4,999 | 4,999 | **2,371** | 0 | 2,628 | 47.43% | 52.57% | 2988.2 | 2591.0 |
| `pmc_patients` | 167,034 | 167,034 | 167,034 | **167,034** | 0 | 0 | 100.0% | 0.0% | 2765.3 | 2491.0 |
| `medqa_usmle` | 14,369 | 14,369 | 14,369 | **14,367** | 0 | 2 | 99.99% | 0.01% | 723.4 | 693.0 |
| `medical_meadow_medqa` | 10,178 | 10,178 | 10,178 | **10,178** | 0 | 0 | 100.0% | 0.0% | 909.3 | 881.0 |
| `symptom2disease` | 1,200 | 1,200 | 1,200 | **1,153** | 0 | 47 | 96.08% | 3.92% | 171.1 | 169.0 |
| `chatdoctor_healthcaremagic` | 112,156 | 112,156 | 112,156 | **111,976** | 0 | 180 | 99.84% | 0.16% | 421.1 | 352.0 |
| `chatdoctor_icliniq` | 7,321 | 7,321 | 7,321 | **7,321** | 0 | 0 | 100.0% | 0.0% | 453.1 | 347.0 |
| `neiss` | 7,326,429 | 7,326,429 | 7,315,729 | **7,137,339** | 10,700 | 178,390 | 97.42% | 2.44% | 87.6 | 78.0 |
| `nhamcs_ed` | 50,548 | 50,548 | 50,548 | **41,509** | 0 | 9,039 | 82.12% | 17.88% | 94.6 | 88.0 |
| `fedmml_ed_triage` | 87,234 | 87,234 | 87,234 | **84,177** | 0 | 3,057 | 96.5% | 3.5% | 255.5 | 251.0 |
| `kaggle_medical_triage` | 2 | 2 | 2 | **0** | 0 | 2 | 0.0% | 100.0% | 0.0 | 0.0 |
| `l3cube_code_mixed` | 44,455 | 44,455 | 44,455 | **43,924** | 0 | 531 | 98.81% | 1.19% | 151.2 | 139.0 |
| `meddialog_en` | 1 | 1 | 1 | **1** | 0 | 0 | 100.0% | 0.0% | 23.0 | 23.0 |

## 2. Department & Specialty Supervision Distribution

| Department / Specialty | Row Count | Percentage |
|-----------------------|-----------|------------|
| `ORTHO` | 2,217,202 | 29.09% |
| `None` | 1,942,787 | 25.49% |
| `ENT_OPHTHALMO` | 1,800,442 | 23.62% |
| `PEDS` | 925,896 | 12.15% |
| `NEURO` | 529,638 | 6.95% |
| `ED` | 125,703 | 1.65% |
| `CARDIO_PULM` | 77,036 | 1.01% |
| `GEN_MED` | 1,202 | 0.02% |
| `SURGERY` | 988 | 0.01% |
| `RENAL_URO` | 228 | 0.00% |
| `GI` | 120 | 0.00% |
| `PSYCH` | 51 | 0.00% |
| `ONCOLOGY_HEME` | 31 | 0.00% |
| `OBGYN` | 26 | 0.00% |

## 3. Language Distribution

| Language Code | Row Count | Percentage |
|---------------|-----------|------------|
| `en` | 7,577,426 | 99.42% |
| `hi-en` | 43,924 | 0.58% |

## 4. Largest Data Loss Sources

### Largest Deduplication Losses (Stage 4)

- `neiss`: **178,390** duplicate rows removed
- `nhamcs_ed`: **9,039** duplicate rows removed
- `fedmml_ed_triage`: **3,057** duplicate rows removed
- `mtsamples`: **2,628** duplicate rows removed
- `l3cube_code_mixed`: **531** duplicate rows removed

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
- `no duplicate IDs`: **CHECKED (IDs assigned per shard)**
- `no duplicate raw_text after export`: **CHECKED (exact duplicate texts deduplicated in Stage 4)**
