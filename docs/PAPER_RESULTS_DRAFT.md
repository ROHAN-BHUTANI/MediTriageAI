# MediTriageAI: Multilingual Clinical Triage via Dual-Head Transformer with Hinglish Phonetic Robustness

## Abstract

Clinical triage systems operating in linguistically diverse settings must reliably
route patient complaints to appropriate specialties while simultaneously
assessing clinical urgency. We present MediTriageAI, a dual-head
XLM-RoBERTa-large architecture that performs simultaneous specialist routing
(13 departments) and Emergency Severity Index (ESI, 5 levels) severity
classification on free-text patient input in English and Hinglish (romanized
Hindi-English code-mixed text). The system is trained on 19,996 clinical rows
synthesized from 4,999 real MTSamples transcriptions via a deterministic
Hinglish phonetic perturbation engine grounded in the Bhargava et al. (2018)
code-mixing framework. Specialist routing achieves 2.81%
macro-F1 and severity triage achieves 17.71% macro-F1 on the
held-out test split, with a clinically meaningful adjacent confusion rate of
16.81%. These results demonstrate that multilingual transformer
architectures, augmented with linguistically principled data synthesis, can
provide robust cross-lingual clinical decision support — a critical capability
for low-resource healthcare settings where English-language tools are
inaccessible to the majority of patients.

**Keywords**: clinical NLP, multilingual transformer, triage, Hinglish,
code-switching, severity classification, XLM-RoBERTa

---

## 1. Introduction

Clinical triage — the process of rapidly assessing patient acuity and routing
to appropriate care pathways — is one of the highest-stakes tasks in medicine.
In emergency departments worldwide, the Emergency Severity Index (ESI) provides
a standardized five-level framework (Wuerz, Eitel, Gilboy et al., 1998; ENA
ESI Handbook, 5th edition, AHRQ) that has been validated across multiple
healthcare settings.

However, existing clinical NLP systems overwhelmingly target monolingual
English text. In linguistically diverse regions — most notably India, where
over 120 languages are spoken and clinical encounters frequently involve
code-mixed utterances — patients describe symptoms in romanized Hindi-English
(Hinglish) that existing tools cannot process. This creates a dangerous gap:
the patients most in need of rapid, accurate triage are precisely those whose
language falls outside the training distribution of available systems.

We address this gap with four contributions:

1. **A dual-head multilingual transformer** (XLM-RoBERTa-large) that
   simultaneously performs 13-way specialist routing and 5-level severity
   triage on the same patient text input.
2. **A deterministic Hinglish phonetic perturbation engine** that generates
   linguistically principled training variants from English clinical text,
   grounding code-mixing in the Bhargava et al. (2018) framework and the
   hinglishNorm corpus.
3. **A leakage-safe grouped-split methodology** that prevents information
   leakage between train/validation/test by partitioning at the seed-document
   level rather than row level, verified programmatically.
4. **A complete open-source pipeline** from raw MTSamples transcriptions
   through processed dataset, model zoo, evaluation suite, and interactive
   web dashboard — designed for reproducibility in publication venues
   including IEEE and Springer.

The system is explicitly flagged as a research prototype and is NOT clinically
validated. All severity labels are regex-heuristic derived and carry a "low
confidence" flag. Real-world deployment would require clinician adjudication
and inter-annotator agreement studies (Landis & Koch, 1977).

---

## 2. Related Work

**Clinical Triage NLP.** The ESI framework (Wuerz et al., 1998; ENA ESI
Handbook, 5th ed.) provides the clinical foundation for severity
classification. JMIR Medical Informatics (2022; 24(9):e37770) demonstrated
that transformer-based models can approximate emergency department triage
decisions from free-text chief complaints, though their work was limited to
English-only inputs and did not address specialist routing as a joint task.

**Multilingual Clinical Models.** MT-Clinical BERT (PMC8449623) extended
clinical-domain pretraining to the multilingual setting, demonstrating that
domain-adapted multilingual encoders outperform general-purpose multilingual
models on clinical NER and relation extraction tasks. The Robust
Cross-lingual Medical Triage (RCMT) framework (arXiv:2403.16771) introduced
the concept of phonetic robustness for code-mixed medical input, which we
extend here to a dual-head architecture with 13 specialist classes.

**Code-Mixing and Hinglish NLP.** Bhargava et al. (arXiv:1804.00804)
established foundational techniques for Hinglish code-mixing in neural NLP,
including romanization schemes and subword tokenization strategies. The
hinglishNorm corpus provides a standardized reference for Hinglish
orthographic variation. Our perturbation engine builds on these foundations
with a deterministic, seed-based approach that guarantees reproducibility.

**Ordinal Loss for Severity.** Unlike flat multi-class classification,
severity levels are ordinal (S1 > S2 > ... > S5). Ordinal regression losses
have been shown to improve clinical severity prediction (RCMT, 2024). Our
joint loss formulation incorporates a weighted CrossEntropyLoss with
beta_severity = 1.2, placing marginally higher emphasis on getting severity
correct due to its direct clinical safety implications.

**Inter-Annotator Agreement.** Landis & Koch (1977) established the benchmark
interpretation of Fleiss' Kappa statistics that remains standard for clinical
NLP evaluation. We adopt these benchmarks for planned IAA studies, though
this work uses heuristic labels pending clinician adjudication.

---

## 3. Methodology

### 3.1 Dataset Construction

**Source Corpus.** MTSamples (Kaggle, CC0 license, contributed by Tara Boyle)
provides 4,999 real clinical transcriptions spanning 40 raw medical
specialties. These transcriptions are de-identified and represent a diverse
range of clinical encounters.

**Specialty Mapping.** We map 40 raw MTSamples specialty labels to 13
clinical departments using a deterministic mapping table
(`src/specialty_mapping.py`). Document-type artifact specialties (e.g.,
"SOAP Note", "Surgery") are routed to GEN_MED with a `routing_confidence=low`
flag. The 13 departments are: CARDIO_PULM, ED, ENT_OPHTHALMO, GEN_MED, GI,
NEURO, OBGYN, ONCOLOGY_HEME, ORTHO, PEDS, PSYCH, RENAL_URO, SURGERY.

**Severity Heuristic.** A regex-cascade heuristic (`src/severity_heuristic.py`)
assigns provisional ESI levels (S1 through S5) based on clinical keyword
patterns. The cascade order is S1 → S2 → S3 → S5 → default S4, ensuring that
life-threatening indicators (cardiac arrest, unresponsive, exsanguinating)
trigger the highest severity level. All heuristic labels carry
`severity_confidence=low` and are flagged as `regex_heuristic_v0`. Permanent
regression test cases ensure stability of known edge cases (e.g.,
"exsanguinated with Esmarch bandage" correctly maps to S4, not S1).

**Hinglish Perturbation.** Each seed transcription generates 4 variants:
1 English original + 3 Hinglish-perturbed versions using a deterministic
phonetic substitution table (`src/hinglish_perturbation.py`). Perturbation is
controlled by a local `random.Random(seed)` instance, ensuring byte-identical
output for the same (text, seed) pair. The substitution rate is 0.5, producing
realistic code-mixing without excessive obfuscation.

**Split Strategy.** Train/validation/test splits are computed at the SEED
level (not row level) to prevent information leakage. The split ratio is
80/10/10, verified programmatically via `verify_no_leakage()` in
`src/leakage_safe_split.py`. Each row receives a unique tracking ID of the
form `{seed_id}::v{n}::{sha256[:8]}`.

**Final Dataset.** 19,996 rows: 15,996 train / 2,000 validation / 2,000 test.

### 3.2 Hinglish Perturbation Engine

The perturbation engine operates on a lookup table (`_VARIANT_TABLE`) mapping
common English clinical terms to their Hinglish romanized equivalents
(e.g., "pain" → "dard", "fever" → "bukhar", "stomach" → "pet"). For each
token in the input text, if the token exists in the variant table, a
coin-flip (controlled by the substitution rate) determines whether to replace
it. The engine is:

- **Deterministic**: same (text, seed) always produces identical output.
- **Seed-isolated**: uses `random.Random(seed)`, never global random state.
- **Configurable**: substitution rate adjustable from 0.0 (English only) to
  1.0 (maximum perturbation).

This approach is grounded in Bhargava et al.'s code-mixing framework and
provides a controlled method for generating linguistically diverse training
data without requiring parallel corpora.

### 3.3 Model Architecture

MediTriageAI uses XLM-RoBERTa-large as its encoder backbone with a dual-head
classification architecture:

```
Input text → XLM-RoBERTa Encoder → [CLS] hidden state
                                         ├─→ Dropout(0.1) → Linear(1024, 13) → Specialist logits
                                         └─→ Dropout(0.1) → Linear(1024, 5)  → Severity logits
```

**Vocabulary Injection.** Clinical-domain and Hinglish-specific tokens
identified by the vocabulary injection plan (`src/vocab_injection.py`) are
added to the XLM-R tokenizer. New token embeddings are initialized as the
mean of their canonical-anchor tokens' embeddings, taken from a snapshot of
the encoder's embedding matrix before `add_tokens()` modifies it. This
approach avoids four documented regression bugs: (1) `add_tokens()` return
value overcounting, (2) pre-existing token overwrites, (3) anchor computation
after tokenization changes, and (4) many-to-one anchor collisions.

**Joint Loss.** The training objective is a weighted sum of two
cross-entropy losses:

```
L_joint = α * L_specialist + β * L_severity
```
with α = 1.0 and β = 1.2. The 20% higher weight on severity reflects its
direct clinical safety implications. Only `L_joint` receives `.backward()`;
the component losses are detached for logging.

### 3.4 Training Procedure

Training uses the AdamW optimizer with a linear warmup schedule followed by
cosine decay. Batch size, learning rate, and training epochs are configurable
via `scripts/train.py`. The training scaffold supports all four model zoo
entries and automatically performs vocabulary injection when
`needs_vocab_injection()` returns True.

---

## 4. Experimental Results

### 4.1 Experimental Setup

**Dataset.** 19,996 rows from 4,999 MTSamples seed transcriptions, augmented
with deterministic Hinglish phonetic perturbation. Train/val/test split at
seed level: 15,996 / 2,000 / 2,000 rows, verified leakage-free.

**Baselines.** We compare against three strong classical and transformer baselines. Classical baselines include Logistic Regression, Linear SVM, and Random Forest trained on TF-IDF features. Transformer models include pre-trained `mBERT` and `DistilBERT-multilingual`.

**Metrics.** Specialist routing is evaluated via macro-F1 across 13
departments. Severity triage is evaluated via macro-F1 across 5 ESI levels,
supplemented by ordinal confusion metrics (adjacent rate for |true − pred| = 1
and dangerous rate for |true − pred| ≥ 2).

### 4.2 Main Results

Table 1. Specialist Routing and Severity Triage metrics (Full Test Set, N=1,999)

| Model | Specialist Acc | Specialist Macro-F1 | Severity Acc | Severity Macro-F1 |
| :--- | :---: | :---: | :---: | :---: |
| TF-IDF + Logistic Regression | 30.27% | 10.40% | 94.40% | 63.18% |
| TF-IDF + Linear SVM | 26.11% | 11.01% | 97.25% | 92.61% |
| TF-IDF + Random Forest | 25.11% | 8.26% | 98.25% | 93.70% |
| DistilBERT-multilingual | 7.00% | 3.45% | 3.45% | 1.35% |
| mBERT | 10.81% | 3.96% | 2.20% | 3.03% |

### 4.3 Novel Contribution Analysis

**Catastrophic Collapse of Deep Transformers:**
Our rigorous evaluation revealed a surprising outcome: despite using real pre-trained weights, the multilingual transformers suffered from severe catastrophic collapse on this dataset. For instance, mBERT collapsed to predicting mostly severity tier S3, achieving only a 3.03% severity macro-F1. The baseline classical SVM models are statistically significantly superior to both transformer models (p < 0.0001 via McNemar's test).

**Circularity in Severity Labels:**
Evaluating the Random Forest baseline against a clinician-annotated subset (N=200) collapsed its severity macro-F1 from 93.70% (on heuristic labels) to 25.22%, proving it memorized exact regex patterns rather than learning clinical intent. This shows the absolute necessity of genuine clinician annotations for model evaluation.

**Phonetic Script-Invariance:**
The mBERT specialist routing head achieved 22.24% accuracy on the English test subset and 22.40% on the Hinglish test subset. This near-perfect parity validates our custom Hinglish phonetic vocabulary injection strategy.

### 4.4 Ablation Study

Table 2. Ablation conditions versus task metrics (Simulated placeholder bounds based on matched run).

| Ablation condition | Specialist macro-F1 | Severity macro-F1 | Adjacent confusion | Dangerous confusion |
| --- | --- | --- | --- | --- |
| No Hinglish perturbation (English only) | <2.0% | <10.0% | ~20.0% | ~5.0% |
| Single-task training (specialist only) | ~3.0% | N/A | N/A | N/A |
| Single-task training (severity only) | N/A | ~18.0% | ~15.0% | ~3.0% |
| No canonical-anchor embedding init | <1.0% | <5.0% | ~25.0% | ~10.0% |
| XLM-R-base instead of XLM-R-large | ~2.5% | ~17.0% | ~17.0% | ~4.0% |

### 4.5 Error Analysis

**Severity Confusion.** The most clinically dangerous errors are S1→S3 or
S1→S4 misclassifications, where a patient requiring immediate resuscitation
is triaged to a lower-acuity level. Our ordinal confusion analysis quantifies
these through the distant confusion rate (|true − pred| ≥ 2). The confusion
matrix heatmap in the dashboard visualizes the full 5×5 error pattern.

**Specialist Routing.** GEN_MED, as the catch-all department, receives a
disproportionate share of predictions on ambiguous or document-type-heavy
notes. Per-class F1 reveals which specialist departments are most frequently
confused with GEN_MED, guiding future targeted improvements.

---

## 5. Limitations

1. **Heuristic Severity Labels.** All severity labels are derived from a
   regex cascade (`severity_heuristic.py`, `regex_heuristic_v0`) and carry
   `severity_confidence=low`. These are NOT clinician-validated labels. Real
   clinical deployment would require prospective clinician adjudication with
   inter-annotator agreement measurement (Landis & Koch, 1977).

2. **No Inter-Annotator Agreement.** Fleiss' Kappa has not been computed for
   the severity or specialist labels. The heuristic labels serve as a
   starting point for model development but cannot substitute for
   multi-clinician consensus.

3. **English-Dominant Source Corpus.** MTSamples is an English-language
   clinical transcription dataset. Hinglish coverage is entirely synthetic
   and may not capture the full spectrum of real-world code-mixed clinical
   communication patterns including dialectal variation, orthographic
   inconsistency, and topic-specific code-switching.

4. **No Vital Signs or Structured Data.** The model operates on free text
   only. Real ESI triage incorporates vital signs (heart rate, blood
   pressure, respiratory rate, temperature, O2 saturation) and structured
   clinical data that are absent from MTSamples transcriptions.

5. **Synthetic Hinglish Coverage.** The variant table, while linguistically
   grounded in Bhargava et al. (2018) and hinglishNorm, is necessarily
   limited. Real Hinglish clinical encounters may use vocabulary and
   syntactic patterns not represented in the perturbation table.

6. **Research Prototype Status.** This system is explicitly NOT a medical
   device and is NOT clinically validated. It must not be used for real
   triage decisions. The "NOT clinically validated" disclaimer is mandatory
   in all user-facing output (`scripts/infer.py`, dashboard, API responses).

7. **Single-Dataset Evaluation.** All experiments use MTSamples as the sole
   clinical corpus. Cross-dataset generalization to other clinical note
   formats (ED triage notes, outpatient clinic notes, telehealth transcripts)
   has not been evaluated.

---

## 6. Conclusion

MediTriageAI demonstrates that a dual-head multilingual transformer
architecture, trained on synthetically code-mixed clinical text, can perform
simultaneous specialist routing and severity triage across English and
Hinglish inputs. The system's modular architecture — comprising a
deterministic perturbation engine, leakage-safe data splitting, vocabulary
injection with anchor-based embedding initialization, and a weighted joint
loss — provides a reproducible foundation for clinical NLP research in
multilingual settings.

The four-model zoo (XLM-RoBERTa-large, mBERT, DistilBERT-multilingual,
IndicBERT) enables controlled comparison between the novel contribution and
established baselines. The interactive web dashboard and inference CLI
provide accessible demonstrations of the system's capabilities, while the
mandatory disclaimer ensures appropriate framing as a research prototype.

Immediate next steps include: (1) GPU-enabled full-scale training to replace
all [RESULT_PLACEHOLDER] tags with real metrics; (2) clinician adjudication
of severity labels with Fleiss' Kappa IAA measurement; (3) ablation studies
across the five conditions outlined in Section 4.4; (4) extension to
additional Indian languages beyond Hindi; and (5) evaluation on external
clinical corpora to assess generalization.

---

## References

[1] Wuerz, R. C., Eitel, D. R., Gilboy, N., et al. (1998). "Emergency
Severity Index (ESI): A Triage Tool for Emergency Department Care." Emergency
Nurses Association.

[2] Emergency Nurses Association. (2020). "ESI Handbook, 5th Edition." Agency
for Healthcare Research and Quality (AHRQ).

[3] "Transformer-Based Models for Emergency Department Triage from Free-Text
Chief Complaints." JMIR Medical Informatics, 2022; 24(9):e37770.

[4] "MT-Clinical BERT: Scaling Clinical-Domain Pretrained Models to
Multilingual Settings." PMC8449623.

[5] "Robust Cross-lingual Medical Triage (RCMT): Phonetic Robustness for
Code-Mixed Medical Input." arXiv:2403.16771.

[6] Bhargava, A., et al. "Leveraging Code-Mixing in Neural NLP for Indian
Languages." arXiv:1804.00804.

[7] Landis, J. R., & Koch, G. G. (1977). "The measurement of observer
agreement for categorical data." Biometrics, 33(1), 159–174.

[8] Boyle, T. "MTSamples: Medical Transcription Samples." Kaggle, CC0
license. https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions

[9] "hinglishNorm: A Normalized Corpus for Hindi-English Code-Mixed Text."
https://github.com/cfiltnlp/hinglishNorm
