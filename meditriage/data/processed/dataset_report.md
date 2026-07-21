# E-PATH-CO-REASON Dataset Audit Report

## 1. Dataset Overview & Size Metrics
- **Total Samples**: 19996
- **Missing Samples (NaN text)**: 33
- **Empty / Whitespace-only Texts**: 0
- **Duplicate Text Instances**: 3725
- **Label Mismatches (Same text, different labels)**: 5795
- **Invalid Specialist Labels**: 0
- **Invalid Severity Labels**: 0

---

## 2. Label & Language Distributions

### Specialist Classes
| Class Code | Samples | Percentage |
| :--- | :---: | :---: |
| GEN_MED | 6244 | 31.23% |
| SURGERY | 4520 | 22.60% |
| ORTHO | 1748 | 8.74% |
| CARDIO_PULM | 1568 | 7.84% |
| NEURO | 1268 | 6.34% |
| GI | 1032 | 5.16% |
| RENAL_URO | 956 | 4.78% |
| ENT_OPHTHALMO | 868 | 4.34% |
| OBGYN | 640 | 3.20% |
| ONCOLOGY_HEME | 360 | 1.80% |
| ED | 300 | 1.50% |
| PEDS | 280 | 1.40% |
| PSYCH | 212 | 1.06% |


### Severity Levels
| Level | Samples | Percentage |
| :--- | :---: | :---: |
| S4 | 18832 | 94.18% |
| S5 | 778 | 3.89% |
| S2 | 201 | 1.01% |
| S3 | 123 | 0.62% |
| S1 | 62 | 0.31% |


### Languages
| Code | Samples | Percentage |
| :--- | :---: | :---: |
| hinglish | 14997 | 75.00% |
| en | 4999 | 25.00% |


---

## 3. Sequence Length Statistics

### Word Count Statistics
- **Maximum Length**: 3029 words
- **Average Length**: 129.16 words
- **Median Length**: 19.0 words
- **25th / 75th Percentile**: 13.0 / 52.0 words
- **90th / 95th / 99th Percentile**: 466.0 / 675.0 / 1103.1 words


### Token Count Statistics (Sampled)
- **Maximum Length**: 3903 tokens
- **Average Length**: 222.01 tokens
- **Median Length**: 39.0 tokens
- **25th / 75th Percentile**: 29.0 / 68.2 tokens
- **90th / 95th / 99th Percentile**: 836.5 / 1162.8 / 1809.1 tokens
