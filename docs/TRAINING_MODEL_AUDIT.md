# MediTriageAI — Authoritative Training Model Architecture Audit

**Specification Baseline:** `v1.0.0-FROZEN`  
**Audit Date:** `2026-08-16`  
**Inspected Source:** [src/model.py](file:///c:/Users/bhuta/Desktop/MediTriageAI_Data_Engine/src/model.py), [models/](file:///c:/Users/bhuta/Desktop/MediTriageAI_Data_Engine/models/)

---

## 1. Ground-Truth Model Architecture Specification

All values below were extracted directly from the authoritative Python source code and runtime inspection of `MediTriageTransformer`.

| Attribute | Source File & Line | Ground-Truth Value | Description / Constraint |
|---|---|---|---|
| **Backbone Architecture** | `src/model.py:7,41` | `XLMRobertaModel` | Pretrained multilingual transformer backbone (`xlm-roberta-base`) |
| **Tokenizer** | `transformers` | `AutoTokenizer.from_pretrained("xlm-roberta-base")` | SentencePiece BPE tokenizer with 250,002 vocabulary size |
| **Hidden Size ($d_{model}$)** | `encoder.config.hidden_size` | `768` | Transformer representation dimensionality |
| **Transformer Layers** | `encoder.config.num_hidden_layers` | `12` | 12 transformer encoder blocks |
| **Attention Heads** | `encoder.config.num_attention_heads` | `12` | 12 self-attention heads per layer (64 dim per head) |
| **Max Sequence Length** | `config.max_position_embeddings` | `512` | Positional embedding capacity |
| **Dropout Probability** | `src/model.py:41,46` | `0.1` | Dropout applied to `[CLS]` token representation before linear heads |
| **Pooling Strategy** | `src/model.py:66` | `last_hidden_state[:, 0, :]` | First token (`[CLS]` / `<s>`) representation extracted from encoder |
| **Specialist Classification Head** | `src/model.py:47` | `nn.Linear(768, 13)` | 13-class linear classifier for clinical department routing |
| **Severity Classification Head** | `src/model.py:48` | `nn.Linear(768, 5)` | 5-class linear classifier for ESI triage acuity (S1–S5) |
| **Activation Functions** | Backbone config | `GELU` (encoder), Linear (heads) | Raw unconstrained logits output for cross-entropy / Focal Loss |
| **Total Parameter Count** | Runtime inspection | `278,057,490` | Exact parameter count |
| **Trainable Parameters** | Runtime inspection | `278,057,490` | 100% of parameters are trainable |
| **Frozen Parameters** | Runtime inspection | `0` | No frozen layers in baseline multi-task configuration |

---

## 2. Parameter Breakdown

```
MediTriageTransformer
├── encoder (XLMRobertaModel):               278,043,648 params (99.995%)
│   ├── embeddings:                          192,001,536 params
│   └── 12 x Transformer Encoder Layers:      86,042,112 params
├── dropout (nn.Dropout, p=0.1):                       0 params
├── classifier_specialist (nn.Linear, 768 -> 13):      9,997 params (0.0036%)
│   ├── weight: (13, 768) -> 9,984
│   └── bias: (13,) -> 13
└── classifier_severity (nn.Linear, 768 -> 5):         3,845 params (0.0014%)
    ├── weight: (5, 768) -> 3,840
    └── bias: (5,) -> 5
────────────────────────────────────────────────────────────────────────
TOTAL PARAMETERS:                            278,057,490 params
```

---

## 3. Specialist & Severity Head Verifications

1. **Specialist Head:**
   - Number of output classes: **13**
   - Class labels: `CARDIO_PULM`, `ED`, `ENT_OPHTHALMO`, `GEN_MED`, `GI`, `NEURO`, `OBGYN`, `ONCOLOGY_HEME`, `ORTHO`, `PEDS`, `PSYCH`, `RENAL_URO`, `SURGERY`.
2. **Severity Head:**
   - Number of output classes: **5**
   - Class labels: `S1` (Resuscitation), `S2` (Emergent), `S3` (Urgent), `S4` (Less Urgent), `S5` (Non-Urgent).
   - Missing / unlabeled ESI: encoded as `-1`, masked by `FocalLoss(ignore_index=-1)`.
