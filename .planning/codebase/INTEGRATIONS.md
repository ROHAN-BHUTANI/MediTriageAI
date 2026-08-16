# MediTriageAI — External Integrations & Data Sources (INTEGRATIONS.md)

**Generated:** 2026-08-14  
**Repository State:** Frozen Baseline (v1.0.0)

---

## 1. External Datasets & Adapters

MediTriageAI unifies 13+ clinical and biomedical corpora into a standardized schema (`patient_presentation`, `department`, `severity`):

| Source Dataset | Adapter Module | Domain & Content | Acuity / Specialty Extraction |
|:---|:---|:---|:---|
| **MTSamples** | `meditriage.builder.adapters.mtsamples` | De-identified clinical transcriptions across specialties | Mapped to 13 target department classes |
| **PMC-Patients** | `meditriage.builder.adapters.pmc_patients` | Case summaries extracted from PubMed Central articles | Department mapped; severity inferred from case severity |
| **MedDialog (EN)** | `meditriage.builder.adapters.meddialog_en` | Doctor-patient consultation dialogues | Multi-turn conversational medical text |
| **ChatDoctor (HealthcareMagic)** | `meditriage.builder.adapters.chatdoctor_healthcaremagic` | Online Q&A medical patient interactions | Specialty-labeled clinical inquiries |
| **ChatDoctor (iCliniq)** | `meditriage.builder.adapters.chatdoctor_icliniq` | Telemedicine clinical queries | Curated patient complaints |
| **FedMML-ED Triage** | `meditriage.builder.adapters.fedmml_ed_triage` | Emergency Department chief complaints and vitals | Dual-annotated for emergency department and triage level |
| **NHAMCS-ED** | `meditriage.builder.adapters.nhamcs_ed` | National Hospital Ambulatory Medical Care Survey | Structured ESI (Emergency Severity Index) acuity levels |
| **NEISS** | `meditriage.builder.adapters.neiss` | National Electronic Injury Surveillance System | Injury narratives and emergency severity |
| **L3Cube Code-Mixed** | `meditriage.builder.adapters.l3cube_code_mixed` | Hindi-English / Indic multilingual clinical texts | Multilingual and code-switched patient complaints |
| **MedicalMeadow MedQA** | `meditriage.builder.adapters.medical_meadow_medqa` | Clinical medical QA knowledge pairs | Clinical diagnosis scenarios |
| **Symptom2Disease** | `meditriage.builder.adapters.symptom2disease` | Symptom descriptions mapped to diagnoses | Symptom-to-specialty routing mapping |
| **MedQA-USMLE** | `meditriage.builder.adapters.medqa_usmle` | USMLE clinical vignettes | Medical knowledge evaluation benchmark |
| **Kaggle Medical Triage** | `meditriage.builder.adapters.kaggle_medical_triage` | Synthetic triage priority dataset | Acuity baseline mapping |

---

## 2. Pretrained Model Backbones (HuggingFace Hub)

The system loads, wraps, or fine-tunes pretrained backbones via HuggingFace:

- **`xlm-roberta-base` / `xlm-roberta-large`**: Primary cross-lingual backbone for high-capacity multi-task routing.
- **`bert-base-multilingual-cased` (`mBERT`)**: Robust cross-lingual baseline supporting 104 languages.
- **`distilbert-base-multilingual-cased`**: Low-latency, resource-constrained inference candidate.
- **`ai4bharat/indic-bert`**: Specialized Indic language model for Hindi, Tamil, Telugu, and code-mixed clinical text.
- **`SimpleClinicalTokenizer` fallback**: Offline-safe whitespace tokenizer provided in `models/base_model.py` for headless environments without HuggingFace access.

---

## 3. Synthetic Augmentation & LLM Providers

For the 10-stage dataset reconstruction engine (`reconstruction/`) and multilingual expansion (`meditriage/multilingual/`), pluggable LLM backends are supported:

| Provider | Module Path | Integration Method | Purpose |
|:---|:---|:---|:---|
| **OpenAI** | `reconstruction.backends` / `meditriage.multilingual.providers` | `OPENAI_API_KEY` via REST API | Clinical paraphrasing, synthetic case generation, class deficit replenishment |
| **Anthropic** | `reconstruction.backends` / `meditriage.multilingual.providers` | `ANTHROPIC_API_KEY` via REST API | Phenotypic variation and clinical hard negative generation |
| **DeepSeek** | `reconstruction.backends` / `meditriage.multilingual.providers` | `DEEPSEEK_API_KEY` via REST API | Cost-effective high-volume augmentation |
| **Local Transformers** | `reconstruction.llm` | Local PyTorch pipeline | Offline synthetic case generation without API keys |

---

## 4. Hardware & Infrastructure Targets

- **Google Colab (T4 / V100)**: Auto-detected by `scripts/colab_train.py` with memory-optimized batch sizing (`max_length=128`, grad-acc=4).
- **NVIDIA DGX / A100 Clusters**: Managed via `scripts/train_ddp.py` with `torchrun`, NCCL backend, and AMP `bfloat16`/`float16`.
- **Single Workstation / CPU Fallback**: Deterministic CPU fallback with mock tensors and graceful device assignment throughout the test suite.
