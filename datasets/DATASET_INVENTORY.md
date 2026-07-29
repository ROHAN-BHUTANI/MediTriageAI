# MediTriageAI — Dataset Inventory Report

**Generated:** 2026-07-29 09:57:41 UTC

---

## Summary

| Status | Count |
|:---|:---:|
| ✅ Downloaded | 2 |
| ⏭️ Skipped (already present) | 10 |
| ❌ Failed | 1 |
| **Total Attempted** | **13** |

---

## ✅ Successfully Downloaded Datasets

| Dataset | License | Files | Size | Source |
|:---|:---|:---:|:---:|:---|
| **symptom2disease** | Open | 8 | 232,734 bytes | [Link](https://huggingface.co/datasets/NeuronZero/Symptom2Disease) |
| **medqa_usmle** | Open | 12 | 131,776,584 bytes | [Link](https://huggingface.co/datasets/bigbio/med_qa) |

### symptom2disease
- **Source:** https://huggingface.co/datasets/NeuronZero/Symptom2Disease
- **License:** Open
- **Download Date:** 2026-07-29T09:56:25.904221+00:00
- **Local Path:** `C:\Users\bhuta\Desktop\MediTriageAI_Data_Engine\datasets\raw\symptom2disease`
- **Files:** 8
- **Total Size:** 232,734 bytes

### medqa_usmle
- **Source:** https://huggingface.co/datasets/bigbio/med_qa
- **License:** Open
- **Download Date:** 2026-07-29T09:57:41.722171+00:00
- **Local Path:** `C:\Users\bhuta\Desktop\MediTriageAI_Data_Engine\datasets\raw\medqa_usmle`
- **Files:** 12
- **Total Size:** 131,776,584 bytes

---

## ⏭️ Skipped Datasets

| Dataset | Reason |
|:---|:---|
| mtsamples | Already downloaded |
| pmc_patients | Already downloaded |
| meddialog_en | Already downloaded |
| chatdoctor_healthcaremagic | Already downloaded |
| chatdoctor_icliniq | Already downloaded |
| fedmml_ed_triage | Already downloaded |
| nhamcs_ed | Already downloaded |
| neiss | Already downloaded |
| l3cube_code_mixed | Already downloaded |
| medical_meadow_medqa | Already downloaded |

---

## ❌ Failed Downloads

### kaggle_medical_triage
- **Source:** https://www.kaggle.com/datasets/daniilkrasnoproshin/medical-triage-priority-dataset
- **License:** Open
- **Reason:** Kaggle CLI not available or authentication failed. Download manually from: https://www.kaggle.com/datasets/daniilkrasnoproshin/medical-triage-priority-dataset


---

## 🔒 Credentialed Datasets (Require Manual Access)

The following datasets are highly relevant to MediTriageAI but require manual registration, institutional credentials, or data use agreements:

| Dataset | Source | Access Requirement | Relevance |
|:---|:---|:---|:---|
| **MIMIC-IV-ED** | PhysioNet | CITI training + DUA | Gold standard ED triage with ESI scores |
| **MIETIC** | PhysioNet | CITI training + DUA | MIMIC-IV triage instruction corpus for LLMs |
| **eICU** | PhysioNet | CITI training + DUA | Multi-center ICU data with acuity scores |
| **i2b2 NLP Challenges** | DBMI Portal | DUA + institutional affiliation | De-identified clinical notes |
| **n2c2 NLP Challenges** | DBMI Portal | DUA + institutional affiliation | Clinical NLP benchmarks |
| **UK Biobank** | UK Biobank | Institutional access | Large-scale health data |
| **CPRD** | CPRD/MHRA | Institutional access + fee | UK primary care data |

---

## 📋 Acquisition Log

Full acquisition log: `download_logs\acquisition_20260729T095622Z.log`
