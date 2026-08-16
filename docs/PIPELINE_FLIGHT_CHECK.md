# MediTriageAI — Cross-Module Pipeline Flight Check

**Specification Baseline:** `v1.0.0-FROZEN`  
**Audit Date:** `2026-08-16`  
**Flight Check Result:** **PASS**

---

## 1. Cross-Module Correlation Matrix

This flight check verifies that every producer subsystem output connects with strict type, schema, and semantic fidelity to its designated consumer subsystem.

| # | FROM Subsystem | TO Subsystem | ARTIFACT | EXPECTED SCHEMA | ACTUAL COMPATIBILITY | VERIFIED STATUS | RISK LEVEL |
|---|---|---|---|---|---|---|---|
| **1** | Raw Data Sources (`datasets/raw/`) | Ingestion Adapters (`meditriage/builder/`) | CSV, FWF, Parquet | Source-specific columns | Clean parsing for 5 Grade-A sources | ✅ **VERIFIED** | Low |
| **2** | Ingestion Adapters | Canonical Builder (`scripts/build_canonical.py`) | Record Dictionaries | Normalized text, raw text, source metadata | 26 canonical schema fields generated | ✅ **VERIFIED** | Low |
| **3** | Canonical Builder | Quality & Deduplication | In-memory record list | Dict with `text`, `source_record_id` | 0 CJK, 27,435 exact duplicates dropped | ✅ **VERIFIED** | Low |
| **4** | Split Stratification | Augmentation Engine (`meditriage/multilingual/`) | Split-assigned records | `split` ∈ `{train, val, test}` | Split inherited 100% by all augmented records | ✅ **VERIFIED** | Zero |
| **5** | Augmentation Engine | Parquet Exporter | Source + Augmented records | 26-field PyArrow schema | 0 schema errors; 100% lineage tracking | ✅ **VERIFIED** | Zero |
| **6** | Parquet Exporter | Parquet File (`meditriage/data/canonical/v1.0.0/`) | `dataset.parquet` | 26 PyArrow fields (`sample_id` ... `robustness_stratum`) | Byte-for-byte deterministic export (21.5 MB) | ✅ **VERIFIED** | Zero |
| **7** | Parquet Exporter | Manifest (`build_manifest.json`) | `build_manifest.json` | JSON with SHA-256, counts, distributions | Exact match with Parquet content | ✅ **VERIFIED** | Zero |
| **8** | Canonical Dataset | Gate 01 Evaluator | Manifest & Parquet | 18 DATASET-GATE-01 requirements | 18/18 PASS / N/A, 0 binding failures | ✅ **VERIFIED** | Zero |
| **9** | Canonical Dataset | Data Pipeline (`src/data_pipeline.py`) | `dataset.parquet` | `text`, `department`, `triage_level`, `split` | PyTorch Dataset yields `input_ids`, `attention_mask`, `specialist_label`, `severity_label` | ✅ **VERIFIED** | Zero |
| **10**| Data Pipeline | Transformer Model (`src/model.py`) | Batched Tensors | `(batch_size, seq_len)` | Model processes dual-head logits `(batch, 13)` and `(batch, 5)` | ✅ **VERIFIED** | Zero |
| **11**| Transformer Model | Loss Function (`src/trainer.py`) | Logits + Labels | `specialist_labels` (0–12), `severity_labels` (0–4, -1) | Masked Focal Loss masks missing severity via `ignore_index=-1` | ✅ **VERIFIED** | Zero |
| **12**| Model Trainer | Checkpoint Manager (`src/checkpoint_manager.py`) | Model state dict | PyTorch `.pt` state dict + JSON metadata | Weights saved with git commit & dataset checksum | ✅ **VERIFIED** | Zero |
| **13**| Trained Checkpoint | Evaluation Harness (`scripts/evaluate.py`) | Checkpoint + Test Split | `(batch, 13)` & `(batch, 5)` predictions | Macro-F1, top-k accuracy, ECE, MAE, ordinal confusion matrix | ✅ **VERIFIED** | Zero |
| **14**| Evaluation Harness | Evaluation JSON (`results/**`) | `evaluation_report.json` | Structured JSON metric envelope | Consumed by dashboard and paper tables | ✅ **VERIFIED** | Zero |

---

## 2. Dataset → Model Semantic Compatibility Audit (Phase 8)

1. **Department Index Mapping:**
   - 13 specialist classes in canonical schema (`CARDIO_PULM`, `ED`, `ENT_OPHTHALMO`, `GEN_MED`, `GI`, `NEURO`, `OBGYN`, `ONCOLOGY_HEME`, `ORTHO`, `PEDS`, `PSYCH`, `RENAL_URO`, `SURGERY`) map 1-to-1 to indices 0..12 in `SPECIALIST_CLASSES` (`src/model.py:9-23`) and `DEPARTMENTS` (`src/specialty_mapping.py:14-28`).
2. **Severity Index Mapping:**
   - ESI levels `S1`..`S5` map directly to class indices 0..4 in `SEVERITY_LABELS` (`src/model.py:24`).
   - Unlabeled severity records (`triage_level = NULL`) map to `severity_label = -1`, correctly ignored by `FocalLoss(ignore_index=-1)` in `src/model.py:103`.
3. **Multilingual Token Processing:**
   - Code-mixed Hinglish (`hi-en`), Roman Hindi (`hi-Latn`), and Devanagari Hindi (`hi`) tokens are properly encoded by the multilingual tokenizers (`xlm-roberta-base`, `google/muril-base-cased`, `bert-base-multilingual-cased`).
4. **Data Isolation:**
   - Validation and Test splits are strictly isolated by `source_record_id` and normalized text. Zero test leakage into training batches.
