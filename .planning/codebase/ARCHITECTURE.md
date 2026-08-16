# MediTriageAI — System Architecture (ARCHITECTURE.md)

**Generated:** 2026-08-14  
**Repository State:** Frozen Baseline (v1.0.0)

---

## 1. High-Level System Overview

MediTriageAI is composed of two primary coupled subsystems:
1. **The Data Engineering & Reconstruction Pipeline (`meditriage/builder`, `reconstruction/`, `src/`)**: Normalizes, sanitizes, deduplicates, and balances heterogeneous clinical notes from 13+ sources.
2. **The Multi-Task Modeling & Emergent Reasoning Subsystem (`meditriage/training`, `models/`, `src/trainer.py`)**: Dual-head neural architectures executing simultaneous specialist department routing and severity acuity classification.

```
 Heterogeneous Sources (13+ Datasets)
                   │
                   ▼
┌────────────────────────────────────────────────────────┐
│               DATA ENGINEERING PIPELINE                │
│  Dataset Adapters ──▶ Normalizer ──▶ Deduplicator      │
│  (13 Sources)          (TriageSchema) (Perceptual Hash)│
│                            │                           │
│                            ▼                           │
│                10-Stage Reconstruction Engine          │
│         (Cluster ──▶ Undersample ──▶ Augment/LLM)      │
└────────────────────────────┬───────────────────────────┘
                             │
                             ▼
               Canonical Dataset (CSV / Parquet)
                             │
                             ▼
┌────────────────────────────────────────────────────────┐
│               MULTI-TASK MODELING ENGINE               │
│                                                        │
│               Pretrained Encoder Backbone              │
│       (XLM-RoBERTa / mBERT / IndicBERT / E-PATH)       │
│                            │                           │
│              ┌─────────────┴─────────────┐             │
│              ▼                           ▼             │
│       Specialist Head              Severity Head       │
│        (13 Classes)                 (5 Classes)        │
│              │                           │             │
│              └─────────────┬─────────────┘             │
│                            ▼                           │
│         Masked Joint Multi-Task Focal Loss             │
│            (handles disjoint annotations)              │
└────────────────────────────┬───────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────┐
│             EVALUATION & INFERENCE ENGINE              │
│  - Bootstrap 95% CIs (1,000 resamples)                 │
│  - Calibration Analysis (ECE, MCE, Brier, NLL)         │
│  - Red-Flag Deterministic Rule Fallback                │
│  - FastAPI REST Microservice (`scripts/serve_api.py`)  │
│  - Interactive Dashboard (`dashboard_web/`)            │
└────────────────────────────────────────────────────────┘
```

---

## 2. Multi-Task Classification Architecture

### Model Architecture (`MediTriageTransformer` & `MultiTaskClinicalClassifier`)
The core model wraps a shared Transformer encoder (e.g., XLM-RoBERTa) with two specialized classification heads:
- **Shared Representation**: Encodes tokenized patient presentation $X \in \mathbb{R}^{B \times L}$ into a pooled CLS representation $h_{\text{CLS}} \in \mathbb{R}^{B \times d}$.
- **Specialist Head**: Linear projection $\mathbb{R}^{d} \to \mathbb{R}^{13}$ predicting target medical department:
  `[CARDIO_PULM, ED, ENT_OPHTHALMO, GEN_MED, GI, NEURO, OBGYN, ONCOLOGY_HEME, ORTHO, PEDS, PSYCH, RENAL_URO, SURGERY]`.
- **Severity Head**: Linear projection $\mathbb{R}^{d} \to \mathbb{R}^{5}$ predicting acuity level:
  `[S1: Resuscitation, S2: Emergent, S3: Urgent, S4: Less Urgent, S5: Non-Urgent]`.

### Loss Formulation: Joint Masked Focal Loss
Because distinct datasets may only annotate department, only annotate severity, or annotate both, loss masking is employed:

$$\mathcal{L}_{\text{joint}} = \alpha \cdot \mathcal{L}_{\text{specialist}}(y_{\text{spec}}, \hat{y}_{\text{spec}}) + \beta \cdot \mathcal{L}_{\text{severity}}(y_{\text{sev}}, \hat{y}_{\text{sev}})$$

Where:
- $\mathcal{L}_{\text{specialist}}$ and $\mathcal{L}_{\text{severity}}$ are **Focal Losses** ($\gamma = 2.0$) with class-imbalance weights.
- Targets with `ignore_index = -1` (unannotated labels) are masked with zero gradient backpropagation, enabling co-training on disjoint clinical datasets without label collision.

---

## 3. E-PATH-CO-REASON Architecture Subsystem

Located under `models/emergent_path_triage/`, this advanced module defines the **Emergent Path-Aligned Co-evolutionary Reasoning Network**:
- **DCCF (`dccf.py`)**: Dynamic Clinical Context Fusion across token segments.
- **AMCO (`amco.py`)**: Adaptive Multimodal Clinical Orchestration.
- **DCES (`dces.py`)**: Dynamic Clinical Evidence Synthesis with cosine orthogonality constraints.
- **DCRR (`dcrr.py`)**: Dynamic Clinical Routing & Reasoning using Gumbel-Softmax discrete path routing.
- **CTB (`ctb.py`)**: Clinical Thought Block applying transformer layers for step-by-step reasoning.
- **DCP (`dcp.py`)**: Dynamic Consistency Projection projecting multi-task logits onto an Urgency Manifold.
- **Hooks & Checkpoints (`hooks.py`, `model.py`)**: Schema-verified checkpoint loading with `EmergentPathCheckpointRegistry`.

---

## 4. 10-Stage Dataset Reconstruction Pipeline (`reconstruction/`)

The reconstruction engine provides a deterministic, multi-stage pipeline:
1. **Stage 1 (Load)**: Ingests raw source datasets.
2. **Stage 2 (Clean)**: Sanitizes medical text, strips formatting artifacts and PII.
3. **Stage 3 (Cluster)**: Semantic clustering of patient presentations.
4. **Stage 4 (Diversity)**: Quantifies vocabulary and phenotypic coverage.
5. **Stage 5 (Undersample)**: Controls majority-class overrepresentation.
6. **Stage 6 (Augment)**: Augments minority classes and underrepresented acuity levels.
7. **Stage 7 (Generate)**: Generates synthetic clinical vignettes via LLM backends.
8. **Stage 8 (Merge)**: Combines real and synthetic partitions with provenance tags.
9. **Stage 9 (Shuffle)**: Applies deterministic seeded shuffling.
10. **Stage 10 (Validate)**: Enforces duplication, contradiction, balance, language, and embedding checks.

---

## 5. Statistical Evaluation & Calibration Subsystem (`analysis/`)

- **Bootstrap Confidence Intervals**: Computes 95% empirical bootstrap intervals over 1,000 resamples for Macro-F1, Weighted-F1, Top-1, and Top-3 accuracy.
- **Calibration Metrics**: Evaluates Expected Calibration Error (ECE), Maximum Calibration Error (MCE), Brier Score, and Negative Log-Likelihood (NLL).
- **Statistical Significance**: Performs McNemar's tests and pairwise Cohen's Kappa agreement matrices.
- **Publication Artifacts**: Auto-compiles 300-DPI heatmaps, reliability curves, and LaTeX table summaries (`PAPER_RESULTS_DRAFT.md`).
