# Dataset Audit Report

## PART 1 — Per-source statistics

| source                     |   total_rows |   train_rows |   val_rows |   test_rows |   unique_languages |   department_coverage_pct |   triage_coverage_pct |   pct_department_missing |   pct_triage_missing |
|:---------------------------|-------------:|-------------:|-----------:|------------:|-------------------:|--------------------------:|----------------------:|-------------------------:|---------------------:|
| chatdoctor_healthcaremagic |       111976 |        89382 |      11371 |       11223 |                  1 |                     0     |                0      |                  100     |             100      |
| chatdoctor_icliniq         |         7321 |         5830 |        748 |         743 |                  1 |                     0     |                0      |                  100     |             100      |
| fedmml_ed_triage           |        84177 |        67695 |       8179 |        8303 |                  1 |                   100     |              100      |                    0     |               0      |
| kaggle_medical_triage      |            2 |            1 |          1 |           0 |                  1 |                     0     |              100      |                  100     |               0      |
| l3cube_code_mixed          |        43924 |        35190 |       4401 |        4333 |                  1 |                     0     |                0      |                  100     |             100      |
| meddialog_en               |            1 |            1 |          0 |           0 |                  1 |                     0     |                0      |                  100     |             100      |
| medical_meadow_medqa       |        10178 |         8153 |       1026 |         999 |                  1 |                     0     |                0      |                  100     |             100      |
| medqa_usmle                |        14367 |        11532 |       1487 |        1348 |                  1 |                     0     |                0      |                  100     |             100      |
| mtsamples                  |         2371 |         1909 |        228 |         234 |                  1 |                   100     |                0      |                    0     |             100      |
| neiss                      |      7137339 |      5709599 |     713540 |      714200 |                  1 |                    77.751 |                0      |                   22.249 |             100      |
| nhamcs_ed                  |        41509 |        33168 |       4114 |        4227 |                  1 |                   100     |               68.4695 |                    0     |              31.5305 |
| pmc_patients               |       167034 |       133304 |      16800 |       16930 |                  1 |                     0     |                0      |                  100     |             100      |
| symptom2disease            |         1153 |          921 |        116 |         116 |                  1 |                   100     |                0      |                    0     |             100      |


## PART 2 — Department distribution

### chatdoctor_healthcaremagic
| source                     |   department |   count |   percentage | is_canonical   | is_missing   |
|:---------------------------|-------------:|--------:|-------------:|:---------------|:-------------|
| chatdoctor_healthcaremagic |          nan |  111976 |          100 | False          | True         |

**Findings:**
- **Unknown/Invalid labels:** None
- **Labels mapped to None:** 111976 rows


### chatdoctor_icliniq
| source             |   department |   count |   percentage | is_canonical   | is_missing   |
|:-------------------|-------------:|--------:|-------------:|:---------------|:-------------|
| chatdoctor_icliniq |          nan |    7321 |          100 | False          | True         |

**Findings:**
- **Unknown/Invalid labels:** None
- **Labels mapped to None:** 7321 rows


### fedmml_ed_triage
| source           | department   |   count |   percentage | is_canonical   | is_missing   |
|:-----------------|:-------------|--------:|-------------:|:---------------|:-------------|
| fedmml_ed_triage | ED           |   84177 |          100 | True           | False        |

**Findings:**
- **Unknown/Invalid labels:** None
- **Labels mapped to None:** 0 rows


### kaggle_medical_triage
| source                |   department |   count |   percentage | is_canonical   | is_missing   |
|:----------------------|-------------:|--------:|-------------:|:---------------|:-------------|
| kaggle_medical_triage |          nan |       2 |          100 | False          | True         |

**Findings:**
- **Unknown/Invalid labels:** None
- **Labels mapped to None:** 2 rows


### l3cube_code_mixed
| source            |   department |   count |   percentage | is_canonical   | is_missing   |
|:------------------|-------------:|--------:|-------------:|:---------------|:-------------|
| l3cube_code_mixed |          nan |   43924 |          100 | False          | True         |

**Findings:**
- **Unknown/Invalid labels:** None
- **Labels mapped to None:** 43924 rows


### meddialog_en
| source       |   department |   count |   percentage | is_canonical   | is_missing   |
|:-------------|-------------:|--------:|-------------:|:---------------|:-------------|
| meddialog_en |          nan |       1 |          100 | False          | True         |

**Findings:**
- **Unknown/Invalid labels:** None
- **Labels mapped to None:** 1 rows


### medical_meadow_medqa
| source               |   department |   count |   percentage | is_canonical   | is_missing   |
|:---------------------|-------------:|--------:|-------------:|:---------------|:-------------|
| medical_meadow_medqa |          nan |   10178 |          100 | False          | True         |

**Findings:**
- **Unknown/Invalid labels:** None
- **Labels mapped to None:** 10178 rows


### medqa_usmle
| source      |   department |   count |   percentage | is_canonical   | is_missing   |
|:------------|-------------:|--------:|-------------:|:---------------|:-------------|
| medqa_usmle |          nan |   14367 |          100 | False          | True         |

**Findings:**
- **Unknown/Invalid labels:** None
- **Labels mapped to None:** 14367 rows


### mtsamples
| source    | department    |   count |   percentage | is_canonical   | is_missing   |
|:----------|:--------------|--------:|-------------:|:---------------|:-------------|
| mtsamples | SURGERY       |     988 |    41.6702   | True           | False        |
| mtsamples | GEN_MED       |     762 |    32.1383   | True           | False        |
| mtsamples | RENAL_URO     |     178 |     7.50738  | True           | False        |
| mtsamples | ORTHO         |      78 |     3.28975  | True           | False        |
| mtsamples | NEURO         |      68 |     2.86799  | True           | False        |
| mtsamples | PEDS          |      52 |     2.19317  | True           | False        |
| mtsamples | PSYCH         |      51 |     2.15099  | True           | False        |
| mtsamples | CARDIO_PULM   |      43 |     1.81358  | True           | False        |
| mtsamples | GI            |      41 |     1.72923  | True           | False        |
| mtsamples | ENT_OPHTHALMO |      36 |     1.51835  | True           | False        |
| mtsamples | ONCOLOGY_HEME |      31 |     1.30747  | True           | False        |
| mtsamples | OBGYN         |      26 |     1.09658  | True           | False        |
| mtsamples | ED            |      17 |     0.716997 | True           | False        |

**Findings:**
- **Unknown/Invalid labels:** None
- **Labels mapped to None:** 0 rows


### neiss
| source   | department    |   count |   percentage | is_canonical   | is_missing   |
|:---------|:--------------|--------:|-------------:|:---------------|:-------------|
| neiss    | ORTHO         | 2217029 |     31.0624  | True           | False        |
| neiss    | ENT_OPHTHALMO | 1800160 |     25.2217  | True           | False        |
| neiss    | nan           | 1587986 |     22.249   | False          | True         |
| neiss    | PEDS          |  925844 |     12.9718  | True           | False        |
| neiss    | NEURO         |  529523 |      7.41905 | True           | False        |
| neiss    | CARDIO_PULM   |   76797 |      1.07599 | True           | False        |

**Findings:**
- **Unknown/Invalid labels:** None
- **Labels mapped to None:** 1587986 rows


### nhamcs_ed
| source    | department   |   count |   percentage | is_canonical   | is_missing   |
|:----------|:-------------|--------:|-------------:|:---------------|:-------------|
| nhamcs_ed | ED           |   41509 |          100 | True           | False        |

**Findings:**
- **Unknown/Invalid labels:** None
- **Labels mapped to None:** 0 rows


### pmc_patients
| source       |   department |   count |   percentage | is_canonical   | is_missing   |
|:-------------|-------------:|--------:|-------------:|:---------------|:-------------|
| pmc_patients |          nan |  167034 |          100 | False          | True         |

**Findings:**
- **Unknown/Invalid labels:** None
- **Labels mapped to None:** 167034 rows


### symptom2disease
| source          | department    |   count |   percentage | is_canonical   | is_missing   |
|:----------------|:--------------|--------:|-------------:|:---------------|:-------------|
| symptom2disease | GEN_MED       |     440 |     38.1613  | True           | False        |
| symptom2disease | ENT_OPHTHALMO |     246 |     21.3356  | True           | False        |
| symptom2disease | CARDIO_PULM   |     196 |     16.9991  | True           | False        |
| symptom2disease | ORTHO         |      95 |      8.23938 | True           | False        |
| symptom2disease | GI            |      79 |      6.85169 | True           | False        |
| symptom2disease | RENAL_URO     |      50 |      4.33651 | True           | False        |
| symptom2disease | NEURO         |      47 |      4.07632 | True           | False        |

**Findings:**
- **Unknown/Invalid labels:** None
- **Labels mapped to None:** 0 rows


## PART 3 — Triage distribution

### chatdoctor_healthcaremagic
| source                     |   triage_level |   count | is_valid   |
|:---------------------------|---------------:|--------:|:-----------|
| chatdoctor_healthcaremagic |            nan |  111976 | True       |


### chatdoctor_icliniq
| source             |   triage_level |   count | is_valid   |
|:-------------------|---------------:|--------:|:-----------|
| chatdoctor_icliniq |            nan |    7321 | True       |


### fedmml_ed_triage
| source           |   triage_level |   count | is_valid   |
|:-----------------|---------------:|--------:|:-----------|
| fedmml_ed_triage |              1 |     921 | True       |
| fedmml_ed_triage |              2 |   16347 | True       |
| fedmml_ed_triage |              3 |   39622 | True       |
| fedmml_ed_triage |              4 |   22424 | True       |
| fedmml_ed_triage |              5 |    4863 | True       |


### kaggle_medical_triage
| source                |   triage_level |   count | is_valid   |
|:----------------------|---------------:|--------:|:-----------|
| kaggle_medical_triage |              2 |       1 | True       |
| kaggle_medical_triage |              4 |       1 | True       |


### l3cube_code_mixed
| source            |   triage_level |   count | is_valid   |
|:------------------|---------------:|--------:|:-----------|
| l3cube_code_mixed |            nan |   43924 | True       |


### meddialog_en
| source       |   triage_level |   count | is_valid   |
|:-------------|---------------:|--------:|:-----------|
| meddialog_en |            nan |       1 | True       |


### medical_meadow_medqa
| source               |   triage_level |   count | is_valid   |
|:---------------------|---------------:|--------:|:-----------|
| medical_meadow_medqa |            nan |   10178 | True       |


### medqa_usmle
| source      |   triage_level |   count | is_valid   |
|:------------|---------------:|--------:|:-----------|
| medqa_usmle |            nan |   14367 | True       |


### mtsamples
| source    |   triage_level |   count | is_valid   |
|:----------|---------------:|--------:|:-----------|
| mtsamples |            nan |    2371 | True       |


### neiss
| source   |   triage_level |   count | is_valid   |
|:---------|---------------:|--------:|:-----------|
| neiss    |            nan | 7137339 | True       |


### nhamcs_ed
| source    |   triage_level |   count | is_valid   |
|:----------|---------------:|--------:|:-----------|
| nhamcs_ed |              1 |     444 | True       |
| nhamcs_ed |              2 |    4318 | True       |
| nhamcs_ed |              3 |   14689 | True       |
| nhamcs_ed |              4 |    7857 | True       |
| nhamcs_ed |              5 |    1113 | True       |
| nhamcs_ed |            nan |   13088 | True       |


### pmc_patients
| source       |   triage_level |   count | is_valid   |
|:-------------|---------------:|--------:|:-----------|
| pmc_patients |            nan |  167034 | True       |


### symptom2disease
| source          |   triage_level |   count | is_valid   |
|:----------------|---------------:|--------:|:-----------|
| symptom2disease |            nan |    1153 | True       |


## PART 4 — Adapter validation

**chatdoctor_healthcaremagic**: Raw dataset contains text input. No department or triage. Intentionally leaves labels empty.

**chatdoctor_icliniq**: Raw dataset contains text input. No department or triage. Intentionally leaves labels empty.

**fedmml_ed_triage**: Raw dataset contains clinical notes/vitals and 'esi_level' triage. Department hardcoded to 'ED'. Labels correctly extracted.

**kaggle_medical_triage**: Raw dataset contains text and 'label' (high/low). Department is not present (intentionally None). Triage is mapped to 2 and 4. Labels correctly extracted.

**l3cube_code_mixed**: Raw dataset contains tokenized text. No department or triage. Intentionally leaves labels empty.

**meddialog_en**: Raw dataset contains dialogues. No department or triage. Intentionally leaves labels empty.

**medical_meadow_medqa**: Raw dataset contains QA text. No department or triage. Intentionally leaves labels empty.

**medqa_usmle**: Raw dataset contains questions. No department or triage. Intentionally leaves labels empty.

**mtsamples**: Raw dataset contains text and 'medical_specialty'. Maps specialty to 'department'. No triage. Labels correctly extracted.

**neiss**: Raw dataset contains narrative, age. Triage is not present. Adapter uses heuristics to derive department (weak labels).

**nhamcs_ed**: Raw dataset contains age, sex, reasons for visit, and 'IMMEDR' for triage. Department hardcoded to 'ED'. Triage logic produces floats instead of strings during pandas aggregation (causing '1.0', '2.0' etc). Fails triage schema validation due to float formatting.

**pmc_patients**: Raw dataset contains patient notes. No department or triage. Intentionally leaves labels empty.

**symptom2disease**: Raw dataset contains text and label. Adapter maps label to 'raw_medical_specialty' but hardcodes 'department_code' to 'UNKNOWN' and completely omits the 'department' key required by the schema. Failing to populate 'department'.

## PART 5 — Supervision matrix

| dataset                    | department_supervision   | severity_supervision   | language   | raw_clinical_text   | synthetic   | real   | code_switched   | weak_labels   | strong_labels   |
|:---------------------------|:-------------------------|:-----------------------|:-----------|:--------------------|:------------|:-------|:----------------|:--------------|:----------------|
| chatdoctor_healthcaremagic | No                       | No                     | en         | Yes                 | No          | Yes    | No              | No            | No              |
| chatdoctor_icliniq         | No                       | No                     | en         | Yes                 | No          | Yes    | No              | No            | No              |
| fedmml_ed_triage           | Yes                      | Yes                    | en         | Yes                 | Yes         | No     | No              | No            | Yes             |
| kaggle_medical_triage      | No                       | Yes                    | en         | Yes                 | No          | Yes    | No              | No            | Yes             |
| l3cube_code_mixed          | No                       | No                     | hi-en      | Yes                 | No          | Yes    | Yes             | No            | No              |
| meddialog_en               | No                       | No                     | en         | Yes                 | No          | Yes    | No              | No            | No              |
| medical_meadow_medqa       | No                       | No                     | en         | Yes                 | No          | Yes    | No              | No            | No              |
| medqa_usmle                | No                       | No                     | en         | Yes                 | No          | Yes    | No              | No            | No              |
| mtsamples                  | Yes                      | No                     | en         | Yes                 | No          | Yes    | No              | No            | Yes             |
| neiss                      | Yes                      | No                     | en         | Yes                 | No          | Yes    | No              | Yes           | No              |
| nhamcs_ed                  | Yes                      | Yes                    | en         | Yes                 | No          | Yes    | No              | No            | Yes             |
| pmc_patients               | No                       | No                     | en         | Yes                 | No          | Yes    | No              | No            | No              |
| symptom2disease            | Yes                      | No                     | en         | Yes                 | No          | Yes    | No              | No            | Yes             |


## PART 6 — Recommendation

**chatdoctor_healthcaremagic**: KEEP FOR PRETRAINING ONLY
- Contains clean raw clinical text but intentionally provides no task supervision.

**chatdoctor_icliniq**: KEEP FOR PRETRAINING ONLY
- Contains clean raw clinical text but intentionally provides no task supervision.

**fedmml_ed_triage**: KEEP FOR BOTH TASKS
- High quality strong supervision for both department (ED) and triage.

**kaggle_medical_triage**: KEEP FOR TRIAGE ONLY
- Provides triage supervision (Emergent/Less Urgent). No department data.

**l3cube_code_mixed**: KEEP FOR PRETRAINING ONLY
- Contains clean raw clinical text but intentionally provides no task supervision.

**meddialog_en**: KEEP FOR PRETRAINING ONLY
- Contains clean raw clinical text but intentionally provides no task supervision.

**medical_meadow_medqa**: KEEP FOR PRETRAINING ONLY
- Contains clean raw clinical text but intentionally provides no task supervision.

**medqa_usmle**: KEEP FOR PRETRAINING ONLY
- Contains clean raw clinical text but intentionally provides no task supervision.

**mtsamples**: KEEP FOR DEPARTMENT ONLY
- Provides good specialty coverage (mtsamples: strong, neiss: weak). No triage data.

**neiss**: KEEP FOR DEPARTMENT ONLY
- Provides good specialty coverage (mtsamples: strong, neiss: weak). No triage data.

**nhamcs_ed**: KEEP FOR BOTH TASKS
- Contains both department (ED) and triage, though adapter requires a bugfix for float formatting.

**pmc_patients**: KEEP FOR PRETRAINING ONLY
- Contains clean raw clinical text but intentionally provides no task supervision.

**symptom2disease**: REMOVE
- Failing adapter. Does not populate department or triage correctly in the final schema.
