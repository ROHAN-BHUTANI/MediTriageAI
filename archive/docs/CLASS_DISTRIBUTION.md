# True Class Distribution (Full Dataset)

This document reports the true label distribution across the entire unfiltered dataset (19,963 valid text rows), completely independent of any pathologically sliced subsets. 

## 1. Specialist Distribution (13 Classes)

| Department Code | Sample Count | Percentage |
| :--- | :--- | :--- |
| **GEN_MED** | 6,242 | 31.27% |
| **SURGERY** | 4,505 | 22.57% |
| **ORTHO** | 1,748 | 8.76% |
| **CARDIO_PULM** | 1,567 | 7.85% |
| **NEURO** | 1,268 | 6.35% |
| **GI** | 1,026 | 5.14% |
| **RENAL_URO** | 954 | 4.78% |
| **ENT_OPHTHALMO** | 866 | 4.34% |
| **OBGYN** | 635 | 3.18% |
| **ONCOLOGY_HEME** | 360 | 1.80% |
| **ED** | 300 | 1.50% |
| **PEDS** | 280 | 1.40% |
| **PSYCH** | 212 | 1.06% |

## 2. Severity Distribution (5 ESI Tiers)

| Severity Level | Sample Count | Percentage |
| :--- | :--- | :--- |
| **S4** | 16,012 | 80.21% |
| **S5** | 2,811 | 14.08% |
| **S2** | 532 | 2.66% |
| **S3** | 396 | 1.98% |
| **S1** | 212 | 1.06% |

---

## Findings & Small-N Flags
- **Zero classes fall below the 100-sample threshold.** The rarest class in both heads is `PSYCH` (212 samples) and `S1` (212 samples). 
- Because all classes have $>200$ examples, simple stratified shuffling (e.g., `train_test_split(stratify=y)`) will be sufficient to ensure every class is adequately represented during training and testing. 
- While no class is fundamentally starved of data, the dataset is still heavily imbalanced overall (e.g., `GEN_MED` has 29x more samples than `PSYCH`, and `S4` has 75x more samples than `S1`). This class imbalance will need to be addressed dynamically (e.g., via class-weighted loss) once the initial slicing bug is resolved.
