# MediTriageAI — Training Data Loader and Pipeline Trace Audit

**Specification Baseline:** `v1.0.0-FROZEN`  
**Audit Date:** `2026-08-16`  
**Audited Modules:** [src/dataset.py](file:///c:/Users/bhuta/Desktop/MediTriageAI_Data_Engine/src/dataset.py), [src/data_pipeline.py](file:///c:/Users/bhuta/Desktop/MediTriageAI_Data_Engine/src/data_pipeline.py), [src/schema.py](file:///c:/Users/bhuta/Desktop/MediTriageAI_Data_Engine/src/schema.py)

---

## 1. End-to-End Pipeline Trace

```
dataset.parquet (Canonical PyArrow Table)
  │
  ▼ [load_split_rows()]
DataFrame split filter (split == "train" | "val" | "test")
  │
  ▼ [validate_and_translate_schema()]
Schema translation & validation (structural checks)
  │
  ▼ [Label Encoding]
Vectorized integer mapping (SPECIALIST_CLASSES: 0..12, SEVERITY_LABELS: 0..4, Missing: -1)
  │
  ▼ [MediTriageDataset.__getitem__]
Tokenizer batching (truncation=True, padding="max_length", max_length=512)
  │
  ▼ [DataLoader Collation]
Batched Tensors:
  ├── input_ids:        torch.Tensor (batch_size, seq_len), dtype=torch.long
  ├── attention_mask:   torch.Tensor (batch_size, seq_len), dtype=torch.long
  ├── labels_specialist:torch.Tensor (batch_size,), dtype=torch.long, values ∈ [-1, 0..12]
  └── labels_severity:  torch.Tensor (batch_size,), dtype=torch.long, values ∈ [-1, 0..4]
  │
  ▼
MediTriageTransformer.forward(input_ids, attention_mask)
```

---

## 2. Invariant & Edge-Case Verification

| Verification Item | Tested Scenario | Observed Behavior | Gate Result |
|---|---|---|---|
| **Column Alignment** | 26 canonical schema columns in Parquet | Successfully mapped to `text`, `label_specialist_id`, `label_severity_id` | **PASS** |
| **Multilingual Text Selection** | Augmented row with Hinglish `text` vs English `raw_text` | Prefers transformed `text` column; feeds actual code-mixed text to tokenizer | **PASS** |
| **Unlabeled Severity Handling** | Rows where `triage_level` is `NULL` (62.1% of train) | Mapped to `label_severity_id = -1`; preserved in batch for department supervision | **PASS** |
| **Split Isolation** | Extraction of `train`, `val`, and `test` splits | 0 cross-split leakage; exact 80/10/10 division maintained | **PASS** |
| **Batch Tensor dtypes** | PyTorch DataLoader output tensors | `input_ids` (int64), `attention_mask` (int64), `labels` (int64) | **PASS** |
| **Attention Mask Validity** | Padded sequence vs unpadded sequence | 1 for active tokens, 0 for `<pad>` tokens | **PASS** |
| **Multi-Worker Safety** | `num_workers > 0` with `pin_memory=True` | Deterministic DataLoader yields with non-blocking CUDA transfers | **PASS** |
