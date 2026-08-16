# MediTriageAI — Dataset License Register

**Specification Baseline:** v1.0.0-FROZEN  
**Gate:** GATE 2 — Dataset Candidate Evaluation  
**Created:** 2026-08-16  
**Status:** AUTHORITATIVE

---

## License Classification System

| Grade | Meaning | Policy |
|---|---|---|
| **A — CLEARLY USABLE** | Public domain, CC0, CC-BY, MIT, Apache 2.0, or US government public-use data with minimal/standard restrictions | May be automatically incorporated |
| **B — ACCESS/DUA REQUIRED** | Requires credentialing, Data Use Agreement, CITI training, or institutional affiliation before access | Must NOT be downloaded/incorporated without completing access requirements; record as RESTRICTED |
| **C — RESTRICTED** | Non-commercial, share-alike, or conditional license that imposes redistribution, modification, or use-case constraints | Requires legal review; may be usable for research only |
| **D — UNKNOWN** | License unclear, missing, or contested; community-uploaded without original publisher verification | Must NOT be used until license is confirmed |
| **E — NOT USABLE** | Scraped without consent, likely PHI, terms of service violations, or copyrighted clinical content | Must be REJECTED |

---

## Existing Repository Datasets

### 1. MTSamples

| Field | Value |
|---|---|
| **Source URL** | https://huggingface.co/datasets/NickyNicky/medical_mtsamples |
| **Original Publisher** | MTSamples.com (transcription service provider) |
| **License** | CC0 1.0 Universal (Public Domain Dedication) |
| **License URL** | https://creativecommons.org/publicdomain/zero/1.0/ |
| **Access Requirements** | None |
| **Redistribution** | Unrestricted |
| **Commercial/Research** | Both permitted |
| **LICENSE GRADE** | **A — CLEARLY USABLE** |

### 2. NEISS (National Electronic Injury Surveillance System)

| Field | Value |
|---|---|
| **Source URL** | https://huggingface.co/datasets/Layered-Labs/neiss-injury-data |
| **Original Publisher** | U.S. Consumer Product Safety Commission (CPSC) |
| **License** | U.S. Government Public Domain |
| **License URL** | https://www.cpsc.gov/cgibin/NEISSQuery/home.aspx |
| **Access Requirements** | None (public-use data) |
| **Redistribution** | Unrestricted (federal government data) |
| **Commercial/Research** | Both permitted |
| **LICENSE GRADE** | **A — CLEARLY USABLE** |

### 3. NHAMCS ED (National Hospital Ambulatory Medical Care Survey)

| Field | Value |
|---|---|
| **Source URL** | https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datasets/NHAMCS/ |
| **Original Publisher** | CDC / NCHS (U.S. Government) |
| **License** | U.S. Government Public-Use Data File |
| **License URL** | https://www.cdc.gov/nchs/data_access/restrictions.htm |
| **Access Requirements** | Implicit data-use agreement (statistical use only, no re-identification, no linking) |
| **Redistribution** | Permitted with conditions (no re-identification) |
| **Commercial/Research** | Both permitted for statistical analysis |
| **LICENSE GRADE** | **A — CLEARLY USABLE** |
| **Notes** | Terms prohibit re-identification attempts. Compliant with our use case (aggregate training). |

### 4. Symptom2Disease

| Field | Value |
|---|---|
| **Source URL** | https://huggingface.co/datasets/NeuronZero/Symptom2Disease |
| **Original Publisher** | Community-contributed (Kaggle origin) |
| **License** | CC0 / Public Domain (Kaggle listing); Apache 2.0 (HuggingFace listing) |
| **License URL** | https://creativecommons.org/publicdomain/zero/1.0/ |
| **Access Requirements** | None |
| **Redistribution** | Unrestricted |
| **Commercial/Research** | Both permitted |
| **LICENSE GRADE** | **A — CLEARLY USABLE** |

### 5. MedQA-USMLE

| Field | Value |
|---|---|
| **Source URL** | https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options |
| **Original Publisher** | Jin et al. (research publication) |
| **License** | MIT License |
| **License URL** | https://github.com/jind11/MedQA (MIT) |
| **Access Requirements** | None |
| **Redistribution** | Permitted with attribution |
| **Commercial/Research** | Both permitted under MIT |
| **LICENSE GRADE** | **A — CLEARLY USABLE** |
| **Notes** | Q&A format; low triage relevance. Useful for medical knowledge augmentation only. |

### 6. Medical Meadow MedQA

| Field | Value |
|---|---|
| **Source URL** | https://huggingface.co/datasets/medalpaca/medical_meadow_medqa |
| **Original Publisher** | Medalpaca project (aggregated from multiple sources) |
| **License** | CC-BY (Creative Commons Attribution) — individual sub-datasets may vary |
| **License URL** | HuggingFace dataset card |
| **Access Requirements** | None |
| **Redistribution** | Permitted with attribution |
| **Commercial/Research** | Both permitted with attribution |
| **LICENSE GRADE** | **A — CLEARLY USABLE** |
| **Notes** | Q&A format; no triage labels, no severity. Limited triage utility. |

### 7. FedMML ED Triage

| Field | Value |
|---|---|
| **Source URL** | https://huggingface.co/datasets/olaflaitinen/fedmml-ed-triage |
| **Original Publisher** | Community-contributed synthetic dataset |
| **License** | HuggingFace community upload; no explicit license file found |
| **License URL** | N/A |
| **Access Requirements** | HuggingFace "accept conditions" prompt |
| **Redistribution** | Unclear — no explicit license |
| **Commercial/Research** | Unclear |
| **LICENSE GRADE** | **D — UNKNOWN** |
| **Notes** | Fully synthetic/LLM-generated. 74% of all severity labels in historical dataset came from this source. License ambiguity is a risk factor. Text provenance = Category C (LLM-generated). |

### 8. ChatDoctor HealthCareMagic

| Field | Value |
|---|---|
| **Source URL** | https://huggingface.co/datasets/lavita/ChatDoctor-HealthCareMagic-100k |
| **Original Publisher** | ChatDoctor project (scraped from HealthCareMagic.com) |
| **License** | **Disputed** — HuggingFace listing shows Apache 2.0, but original project states non-commercial; data scraped from a commercial medical Q&A platform |
| **License URL** | https://github.com/Kent0n-Li/ChatDoctor |
| **Access Requirements** | None for download; legal risk for use |
| **Redistribution** | **High risk** — original data scraped without verified consent |
| **Commercial/Research** | Original project: academic/research only. Commercial use generally prohibited. |
| **LICENSE GRADE** | **E — NOT USABLE** |
| **Notes** | Scraped patient-doctor interactions from a commercial platform. The Apache 2.0 tag on HuggingFace was applied by a third-party uploader, not the original data owner. High legal and ethical risk. |

### 9. ChatDoctor iCliniq

| Field | Value |
|---|---|
| **Source URL** | https://huggingface.co/datasets/lavita/ChatDoctor-iCliniq |
| **Original Publisher** | ChatDoctor project (scraped from iCliniq.com) |
| **License** | Same disputed status as HealthCareMagic |
| **License URL** | https://github.com/Kent0n-Li/ChatDoctor |
| **Access Requirements** | None for download; legal risk for use |
| **Redistribution** | **High risk** |
| **Commercial/Research** | Research only at best; likely not usable |
| **LICENSE GRADE** | **E — NOT USABLE** |
| **Notes** | Same provenance and legal concerns as ChatDoctor HealthCareMagic. Scraped from a commercial telemedicine platform. |

### 10. PMC-Patients

| Field | Value |
|---|---|
| **Source URL** | https://huggingface.co/datasets/zhengyun21/PMC-Patients |
| **Original Publisher** | Zhengyun et al. (research publication) |
| **License** | CC BY-NC-SA 4.0 |
| **License URL** | https://creativecommons.org/licenses/by-nc-sa/4.0/ |
| **Access Requirements** | None |
| **Redistribution** | ShareAlike, non-commercial |
| **Commercial/Research** | **Research only** (non-commercial) |
| **LICENSE GRADE** | **C — RESTRICTED** |
| **Notes** | Non-commercial restriction. Usable for research/academic purposes. Any derivative dataset must use same license. |

### 11. Kaggle Medical Triage

| Field | Value |
|---|---|
| **Source URL** | https://huggingface.co/datasets/sweatSmile/medical-symptom-triage-csv |
| **Original Publisher** | Community-contributed (Kaggle origin) |
| **License** | CC0 / Public Domain (per Kaggle listing) |
| **License URL** | https://creativecommons.org/publicdomain/zero/1.0/ |
| **Access Requirements** | None |
| **Redistribution** | Unrestricted |
| **Commercial/Research** | Both permitted |
| **LICENSE GRADE** | **A — CLEARLY USABLE** |
| **Notes** | Very small (1,112 rows). Uses string-based severity labels (Routine/Urgent/Emergency/Observation) rather than ESI 1–5. |

### 12. L3Cube HingLID (Code-Mixed)

| Field | Value |
|---|---|
| **Source URL** | https://raw.githubusercontent.com/l3cube-pune/code-mixed-nlp/main/L3Cube-HingLID/train.txt |
| **Original Publisher** | L3Cube Pune (academic research group) |
| **License** | MIT License (per GitHub repository) |
| **License URL** | https://github.com/l3cube-pune/code-mixed-nlp/blob/main/LICENSE |
| **Access Requirements** | None |
| **Redistribution** | Permitted with attribution |
| **Commercial/Research** | Both permitted |
| **LICENSE GRADE** | **A — CLEARLY USABLE** |
| **Notes** | Token-level language-identification annotations, not medical content. Value is Hinglish linguistic patterns, not clinical relevance. Not a triage dataset. |

### 13. MedDialog EN

| Field | Value |
|---|---|
| **Source URL** | OpenMed/MedDialog on HuggingFace |
| **Original Publisher** | Zeng et al. (research publication) |
| **License** | Unclear — original paper does not specify; HuggingFace community upload |
| **License URL** | N/A |
| **Access Requirements** | None for download |
| **Redistribution** | Unclear |
| **Commercial/Research** | Unclear |
| **LICENSE GRADE** | **D — UNKNOWN** |
| **Notes** | 25.6% CJK contamination detected in historical build. License status unverified against original publisher. Contains patient-doctor dialogues — provenance and consent status unknown. |

---

## External Candidate Datasets (Not Currently in Repository)

### 14. MIMIC-IV-ED

| Field | Value |
|---|---|
| **Source URL** | https://physionet.org/content/mimic-iv-ed/ |
| **Original Publisher** | MIT Laboratory for Computational Physiology / PhysioNet |
| **License** | PhysioNet Restricted Health Data License 1.5.0 |
| **License URL** | https://physionet.org/content/mimic-iv-ed/2.2/ |
| **Access Requirements** | PhysioNet credentialing, CITI training, institutional affiliation, signed DUA |
| **Redistribution** | **Prohibited** |
| **Commercial/Research** | Research only |
| **LICENSE GRADE** | **B — ACCESS/DUA REQUIRED** |
| **Notes** | Gold-standard ED triage dataset (~425K stays with ESI). Cannot be incorporated without completing credentialing. Recorded for future consideration. |

### 15. MIETIC (MIMIC-IV-Ext Triage Instruction Corpus)

| Field | Value |
|---|---|
| **Source URL** | https://physionet.org/content/mietic/ |
| **Original Publisher** | PhysioNet (2025 release) |
| **License** | PhysioNet Restricted Health Data License |
| **License URL** | https://physionet.org |
| **Access Requirements** | Same as MIMIC-IV-ED |
| **Redistribution** | **Prohibited** |
| **Commercial/Research** | Research only |
| **LICENSE GRADE** | **B — ACCESS/DUA REQUIRED** |

### 16. Triagegeist (Kaggle Synthetic ED Dataset)

| Field | Value |
|---|---|
| **Source URL** | https://www.kaggle.com/datasets/ (search: triagegeist) |
| **Original Publisher** | Community-contributed (Kaggle) |
| **License** | CC0 / Public Domain (per Kaggle listing) |
| **License URL** | https://creativecommons.org/publicdomain/zero/1.0/ |
| **Access Requirements** | Kaggle account |
| **Redistribution** | Unrestricted |
| **Commercial/Research** | Both permitted |
| **LICENSE GRADE** | **A — CLEARLY USABLE** |
| **Notes** | Fully synthetic. Mimics MIMIC-IV-ED distributions. Includes ESI, demographics, vitals, chief complaints. Useful as auxiliary severity-labeled source. Must be marked as Category C (synthetic). |

---

## License Gate Summary

| Dataset | Grade | Final Status |
|---|---|---|
| MTSamples | **A** | ✅ CLEARED |
| NEISS | **A** | ✅ CLEARED |
| NHAMCS ED | **A** | ✅ CLEARED |
| Symptom2Disease | **A** | ✅ CLEARED |
| MedQA-USMLE | **A** | ✅ CLEARED |
| Medical Meadow MedQA | **A** | ✅ CLEARED |
| Kaggle Medical Triage | **A** | ✅ CLEARED |
| L3Cube HingLID | **A** | ✅ CLEARED |
| PMC-Patients | **C** | ⚠️ RESTRICTED (NC-SA) |
| FedMML ED Triage | **D** | ⚠️ UNKNOWN LICENSE |
| MedDialog EN | **D** | ⚠️ UNKNOWN LICENSE |
| ChatDoctor HealthCareMagic | **E** | ❌ REJECTED |
| ChatDoctor iCliniq | **E** | ❌ REJECTED |
| MIMIC-IV-ED | **B** | 🔒 DUA REQUIRED |
| MIETIC | **B** | 🔒 DUA REQUIRED |
| Triagegeist | **A** | ✅ CLEARED (if acquired) |

---

## Governance Rules

1. **Only Grade A datasets may be automatically incorporated** into the canonical pipeline.
2. **Grade B datasets** require the user to complete credentialing independently before they can be considered.
3. **Grade C datasets** may be used for research only; any derivative must carry the same license.
4. **Grade D datasets** must NOT be used until their license is confirmed with the original publisher.
5. **Grade E datasets** are permanently REJECTED and must be quarantined from the canonical build.
