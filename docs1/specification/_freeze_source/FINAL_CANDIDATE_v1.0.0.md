# MediTriageAI Specification Baseline — FINAL CANDIDATE v1.0.0
### Status: **CANDIDATE FOR FREEZE — NOT YET FROZEN**
### This is the single, physically-consolidated document superseding v0.1 → v0.4. Nothing here is `v1.0.0-FROZEN` until you give explicit written approval on the checklist at the end.

---

## 0. HOW TO READ THIS DOCUMENT

Tags used throughout, applied per-statement, never assumed:

- **FACT** — verified directly against repository evidence (file/line citations from the two Antigravity tool-trace audits behind v0.1/v0.3).
- **IMPLEMENTED / PARTIAL / EXPERIMENTAL / MISSING / UNKNOWN** — implementation-status taxonomy, used mainly in the Multilingual Robustness Audit Matrix (Section 6) and anywhere a capability's completeness (not just existence) is being assessed.
- **INTENT** — your stated research/product north star.
- **PROPOSED** — a recommendation still open for approval.
- **DECIDED** — locked by your explicit approval (this version incorporates every DECIDED item through your latest message).
- **CONTRADICTED** — a documentation/paper claim disproven by code evidence.
- **TBD** — deliberately left unset because no defensible basis exists yet to set it (used for thresholds, dataset sizes — never filled with a plausible-sounding placeholder).

Five-way system-state classification (from v0.3), applied throughout: **CURRENT** / **TARGET** / **MIGRATION** / **RESEARCH-EXPERIMENTAL** / **FUTURE-OPTIONAL**.

**Nothing is promoted to FACT without repository evidence in this document.** Where prior sessions' commentary-style input (v0.4's source material) hasn't been independently verified, it remains explicitly UNKNOWN, no matter how specific it reads.

---

## 1. SPEC-01 — PRODUCT & RESEARCH REQUIREMENTS

**Core problem (INTENT):** route free-text patient presentations in English/Hinglish to 1 of 13 specialist departments and 1 of 5 ESI-style severity levels, in a research context, from disjoint, partially-annotated clinical datasets.

**Scope — DECIDED, Track C:** "Full product-grade research/demo system" — polished CLI, API, dashboard, documentation, observability, and reproducibility tooling, built to a high engineering/design bar, in service of generating and communicating trustworthy research results. Explicitly **not** a clinically deployed product, not submitted for regulatory clearance, makes no autonomous-triage claim. Apple-quality is a communication-clarity bar, not permission to imply validation the system doesn't have — a polished screen implying clinical reliability it hasn't earned is a worse outcome than a plain one.

**Users:** researchers/reviewers (primary, INTENT+FACT — matches paper-first framing and advisor/venue context). Nurses/intake staff (future/INTENT-only, not promoted by Track C).

**Non-negotiable requirement (INTENT, hard constraint, cross-cutting):** "NOT clinically validated / not a medical device" disclaimer, byte-identical, on every user-facing surface: CLI, API response envelope, every dashboard view, README, paper limitations section.

**Success criteria:**
1. Novelty paragraph and all headline metrics are programmatically generated from real evaluation output — never hand-typed (FR-METRICS-01).
2. Dashboard/API independently deployable from a documented, reproducible build (SPEC-13), not dependent on one machine's local state.
3. First-time dashboard visitor understands the research-prototype status before seeing any prediction.
4. No official benchmark/training campaign proceeds on a dataset artifact that hasn't passed DATASET-GATE-01 (Section 5).
5. No claim of Hinglish/multilingual "understanding" appears anywhere without being scoped to the specific dimensions actually demonstrated per the Multilingual Robustness Audit Matrix (Section 6).

**In scope (FACT, current repo):** 13-department routing, 5-level ESI severity, English + Hinglish, dual-head transformer classification, MTSamples + auxiliary corpora, 5-backbone model zoo, statistical evaluation infrastructure.

**Out of scope (FACT, must stay out absent an approved CR):** live patient triage, multimodal input, multi-node FSDP, EHR/HL7 write-back.

---

## 2. SPEC-02 — SOFTWARE REQUIREMENTS + ACCEPTANCE CRITERIA

| ID | Requirement | Acceptance Criteria | Status |
|---|---|---|---|
| FR-DATA-01 | Every label carries source-type provenance (`DIRECT`/`MAPPED`/`INFERRED`) | Canonical schema row always has `label_provenance`; test asserts no row missing it | FACT — already the case |
| FR-DATA-02 | EXCLUDED datasets never enter active training without explicit override | Test: loading `l3cube_code_mixed`/`medical_meadow_medqa`/`medqa_usmle` raises by default | UNKNOWN — not confirmed enforced; add as MIGRATION-phase test |
| **FR-DATA-03** | **DATASET-GATE-01** — see full spec, Section 5 | Gate report generated and passing before any training run | **DECIDED, binding, new** |
| FR-TEXT-01 | Every row's text (not just label) carries a text-provenance category: `SOURCE`, `A` (deterministic linguistic augmentation), `B` (rule-based/templated construction), `C` (LLM-generated) | Field present, queryable, never pooled into an anonymous "synthetic" bucket | DECIDED, new (v0.4 taxonomy) |
| FR-MODEL-01 | Model zoo supports ≥2 backbones through one eval harness | `pytest tests/test_model_zoo.py` | FACT, implemented |
| FR-METRICS-01 | Novelty paragraph generated programmatically from evaluation JSON | `src/metrics.py`/equivalent exists; no file contains a hand-authored novelty claim | NOT YET IMPLEMENTED — critical-path build item |
| FR-EVAL-01 | Evaluation config hash computed from actual config, never a placeholder | `"TODO_config_hash"` at `src/evaluation.py:137` removed; replaced by deterministic hash | CURRENT = broken (FACT); TARGET specified in Section 2's FR-EVAL-03 below |
| **FR-EVAL-02** | **Primary model-selection metrics frozen before the final DGX benchmark campaign; no retrospective metric selection** | Metric names appear in a version-controlled pre-registration artifact dated before the benchmark run | **DECIDED** — see Section 9 for the actual metric list |
| **FR-EVAL-03** | Deterministic, canonical configuration hashing mechanism | If `config.get_hash()` (or equivalent in `meditriage/training/config.py`) exists and is deterministic/canonical, use it; if not, implement one as part of the migration. All placeholder/TODO config-hash claims removed from the final system. | **DECIDED** — verification pending during SPEC-03 migration |
| FR-SAFETY-01 | Disclaimer present, byte-identical, across CLI/API/dashboard | Snapshot test asserts presence and exact match | PROPOSED test, not yet confirmed to exist |
| FR-SAFETY-02 | Documentation linter fails build if red-flag mentions lack the mandatory "unaudited deterministic heuristic fallback" phrase nearby | Grep-based CI check | PROPOSED |
| FR-UX-01 | Prediction surfaces show calibration-aware confidence, not bare softmax | UI value sourced from actual ECE/reliability data, not hardcoded | PROPOSED (SPEC-08) |
| FR-API-01 | API separates provenance-derived confidence from model confidence | Distinct response fields; test asserts no conflation | PROPOSED (SPEC-09) |
| FR-OPS-01 | `/version` (or equivalent) reports exact model checkpoint hash + spec version | Test hits endpoint, asserts match to build manifest | PROPOSED (SPEC-13) |
| FR-SEC-01 | No PHI ingestion path anywhere in the system | Architecture review checklist item; no code path accepts identifiable patient data by design | DECIDED as a permanent constraint (SPEC-14) |

**Definition of Done (unchanged convention, hosted formally in SPEC-10):** code merged + test passing + (if touching labels/severity/red-flags/multilingual claims) limitations doc updated in the same PR + no new TODO/placeholder strings introduced.

---

## 3. SPEC-03 — SYSTEM ARCHITECTURE

**`meditriage/` canonicalization — DECIDED, migration not blanket cutover.** Direct-audit FACT: `meditriage/` currently has inbound dependencies on `src/` (`models/base_model.py` subclasses `src.model.MediTriageTransformer`; `meditriage/builder/adapters/mtsamples.py` imports `src.specialty_mapping`; `scripts/train_ddp.py` imports `EmergentTrainer`/`TrainingConfig` from `src.trainer`/`src.config_manager`). Migration proceeds in audited-risk order:

- **Priority 1 (CRITICAL):** `src/model.py`, `src/trainer.py`, `src/dataset.py` (no `meditriage/` equivalent exists — must be built, not ported), `src/schema.py`. Parity requirement: exact numerical logit parity on a frozen checkpoint/test slice (model/trainer); step-by-step training trajectory comparison, not just final metric (trainer); identical accept/reject behavior on the same record set (schema).
- **Priority 2 (HIGH):** `src/specialty_mapping.py`, `src/config_manager.py`, `src/vocab_injection.py`, `src/hinglish_perturbation.py`, `src/metrics.py`, `src/evaluation.py`, `src/data_pipeline.py`.
- **Priority 3 (MEDIUM):** `src/checkpoint_manager.py`, `src/sampling.py`, `src/profiler.py`.
- **Priority 4 (LOW, LEGACY BUT ACTIVE):** `src/dashboard.py`, `src/calibration.py`, `src/dataset_adapters.py`, `src/dataset_registry.py`, `src/data_ingestion.py`, `src/transforms/hinglish_perturbation.py`.
- **No action / safe to delete (DEAD/UNREFERENCED, FACT, zero inbound imports):** `src/clinical_safety_validator.py`, `src/diversity_scorer.py`, `src/duplicate_validator.py`, `src/experiment_manager.py`, `src/leakage_safe_split.py`, `src/registry.py`, `src/severity_heuristic.py`, `src/transformation_base.py`, `src/transforms/*.py` (12 files). Confirm zero references immediately before deletion.
- **TEST-ONLY:** `src/explainability.py` — decide to port (if wanted for SPEC-08's saliency UI) or retire with its tests.

**Hard gate (DECIDED):** Priority-1 modules must reach verified numerical parity before the first official frozen-spec paper results run. Any `src/`-sourced number used before full migration completes must be visibly footnoted "(legacy harness)."

**Execution environment boundary (DECIDED, binding governance, not just process color):**
- **LOCAL/Antigravity:** spec management, implementation, refactoring, UI/API construction, unit + integration tests, GSD planning/execution, Ralph bounded implementation, CodeRabbit review, git/version control.
- **DGX:** canonical ML environment; dataset preparation/acquisition where appropriate; expensive training; full evaluation; long-running/full test campaigns; multi-GPU execution; reproducibility runs; final benchmark generation; final research artifacts.
- DGX has no authority to silently redefine the frozen spec. A DGX-discovered problem is a Change Request, filed identically to a LOCAL-discovered one.
- Every DGX-produced artifact (training run, evaluation report, benchmark) is tagged with both the frozen spec version it was executed against AND the dataset checksum it consumed (this ties directly to DATASET-GATE-01, Section 5, item 17).

---

## 4. SPEC-04 — DATA ARCHITECTURE & CONTRACTS

**Canonical schema (FACT):** `patient_presentation` (str, min len 5), `department` (int 0–12 or -1), `severity` (int 0–4 or -1).

**Dataset tiering (FACT):**
- CORE: `neiss`, `nhamcs_ed`, `mtsamples`
- AUXILIARY: `pmc_patients`, `chatdoctor_healthcaremagic`, `chatdoctor_icliniq`, `symptom2disease`, `meddialog_en`, `fedmml_ed_triage`, `kaggle_medical_triage`
- EXCLUDED: `l3cube_code_mixed`, `medical_meadow_medqa`, `medqa_usmle`

**Label-quality risk table (carried forward, unchanged substance):**

| Risk | Evidence | Required mitigation |
|---|---|---|
| Regex-inferred specialty labels (4/13 datasets) may correlate with regex-based severity heuristic by construction | Both keyword-based | Report Macro-F1 split by label-provenance tier; never pool without this breakdown |
| Severity coverage ~1.3% of rows, 63% synthetic (`fedmml_ed_triage`) | `dataset_label_provenance.md:196-205` | Severity results always reported with "% synthetic" figure + a sensitivity analysis excluding synthetic rows |
| NEISS is 70% of volume, injury-only | `dataset_governance.md:76-81` | Per-class recall reporting, not just macro-averaged accuracy |
| 512-token truncation may drop late-appearing red flags | `CONCERNS.md:28-30` | Truncation-rate metric reported; flagged as interacting with red-flag safety system |

**Text-provenance taxonomy — DECIDED, new axis, orthogonal to label provenance (FR-TEXT-01):**

| Category | Definition | Known example |
|---|---|---|
| `SOURCE` | Unmodified real clinical text | MTSamples, NEISS, NHAMCS narratives |
| `A` — Deterministic linguistic augmentation | Real text → controlled deterministic transformation (phonetic substitution, ASR-noise simulation, synonym swap) | The Hinglish variant table (`hai`→`hain`/`he`/`hy`, etc. — REPORTED, pending direct verification) |
| `B` — Rule-based/templated construction | Clinical concepts → controlled constructed examples via fixed templates, no LLM | The "offline rule-based provider" (REPORTED, pending verification), distinct from LLM providers |
| `C` — LLM-generated | LLM → generated clinical text | `fedmml_ed_triage` (already FACT, already flagged as 63% of severity-labeled rows) |

**Required action:** retroactively audit every AUXILIARY dataset's augmentation to assign it a text-provenance category — "synthetic" as used in prior spec versions has been ambiguous across A/B/C and must not remain pooled going forward.

---

## 5. DATASET-GATE-01 — CANONICAL TRAINING DATASET GATE (DECIDED, BINDING)

**No official model-training campaign — LOCAL or DGX — may begin until all of the following are satisfied and recorded in a single, versioned dataset-gate report:**

1. Raw source datasets versioned and checksummed.
2. Canonical ingestion complete.
3. Multilingual expansion complete.
4. Hinglish/romanization variation generation complete.
5. Linguistic robustness augmentation complete.
6. Phenotype augmentation complete where enabled.
7. Hard-negative generation complete where enabled.
8. Quality validation passes.
9. Deduplication passes.
10. Train/validation/test leakage audit passes.
11. **Language-distribution report generated** (the direct structural fix for the English-dominance risk in Section 5a below).
12. Class-distribution report generated.
13. Train/validation/test isolation explicitly verified (not merely assumed from the leakage-safe split design).
14. Provenance recorded for every generated sample, using the SPEC-04 text-provenance taxonomy (`SOURCE`/`A`/`B`/`C`).
15. Synthetic/generated-vs-source proportions reported, broken out by taxonomy category, never pooled.
16. Final dataset receives a SHA-256 checksum.
17. Training configuration references that exact checksum.
18. Any DGX training run records the dataset checksum in its output artifacts.

**Binding failure behavior (DECIDED):** the training entrypoint must **fail loudly** — not silently fall back — if the checksum it's configured with doesn't match the dataset actually present on disk. The system must never silently train on an older builder output when a newer canonical dataset has been specified.

**Acceptance criterion for item 11:** the language-distribution report must be generated and reviewed **before** any DGX training run is authorized, not as a post-hoc audit artifact.

### 5a. Immediate language-distribution audit — DECIDED, action authorized, **execution status: PENDING**

You approved this audit. I do not have DGX or live-repository access in this chat session — I cannot execute it myself. What I can do, and have done, is specify exactly what the audit must produce, so it's ready to hand to Antigravity/DGX:

**Required audit output format:**
```
artifact_path: <exact path>
artifact_checksum: <SHA-256>
row_count: <int>
language_counts: { en: <int>, hi_latn: <int>, hi_en_mixed: <int>, ... }
language_percentages: { ... }
specialist_distribution: { CARDIO_PULM: <pct>, ED: <pct>, ... (all 13) }
severity_distribution: { S1: <pct>, ..., S5: <pct> }
provenance_distribution: { SOURCE: <pct>, A: <pct>, B: <pct>, C: <pct> }   # if available
dataset_gate_status: PASS | FAIL, with the specific failing item numbers listed if FAIL
```

**Binding rule (DECIDED, your wording):** the reported ~99.42%-English figure from the prior commentary-style input is assumed **neither true nor false** — it is determined from the actual artifact, once run. If an old training artifact fails DATASET-GATE-01, it is classified **NOT ELIGIBLE** for the final benchmark campaign — full stop, no undocumented rescue attempt, no partial-credit reasoning.

**This audit has not been run in this session.** Its results are UNKNOWN until executed on DGX. Section 20 (DGX Execution Contract) below specifies exactly how to hand this off.

---

## 6. MULTILINGUAL ROBUSTNESS FRAMEWORK + AUDIT MATRIX (DECIDED framework, PER-DIMENSION STATUS MOSTLY UNKNOWN)

**Framework adopted, per your approval, with the explicit caveat that adoption of the framework is not adoption of the claim that all 20 capabilities exist.** For each dimension, the required evidentiary chain is: **generation/source mechanism → training exposure → validation coverage → evaluation metric → acceptance criterion.** A dimension only counts as demonstrated once all five links are shown with evidence — not inferred from the pipeline diagram existing.

| # | Dimension | Status | Generation/Source | Training Exposure | Validation | Evaluation Metric | Acceptance Criterion |
|---|---|---|---|---|---|---|---|
| 1 | Standard English | IMPLEMENTED | SOURCE data (FACT — majority of corpora) | FACT | Standard test split | Existing Macro-F1 | Primary metric (Section 9) |
| 2 | Standard Hindi (Devanagari) | MISSING | Per your project history, prior scope included Devanagari Hindi but current active build focuses on English + Hinglish (romanized) | N/A | N/A | N/A | Explicitly out of current scope per your own project memory — do not claim |
| 3 | Romanized Hindi | PARTIAL | REPORTED deterministic phonetic engine (`src/hinglish_perturbation.py`, FACT that the module exists per direct audit) | UNKNOWN — coverage in actual training runs not confirmed | UNKNOWN | UNKNOWN | Requires dedicated audit |
| 4 | English-Hindi code mixing | PARTIAL | Multilingual expansion pipeline stage referenced in project docs (REPORTED) | UNKNOWN | UNKNOWN | UNKNOWN | Requires dedicated audit |
| 5 | Phonetic transliteration | PARTIAL | Same variant-table mechanism as #3 | UNKNOWN | UNKNOWN | UNKNOWN | Requires dedicated audit |
| 6 | Common spelling variation | UNKNOWN | Not independently verified this session | UNKNOWN | UNKNOWN | UNKNOWN | Requires dedicated audit |
| 7 | Informal chat spelling | UNKNOWN | — | — | — | — | Requires dedicated audit |
| 8 | Abbreviations | UNKNOWN | — | — | — | — | Requires dedicated audit |
| 9 | Clinical shorthand | UNKNOWN | — | — | — | — | Requires dedicated audit |
| 10 | ASR-like transcription noise | PARTIAL | REPORTED "ASR noise" transform in the 10-style linguistic variation engine | UNKNOWN | UNKNOWN | UNKNOWN | Requires dedicated audit |
| 11 | Synonym variation | PARTIAL | REPORTED synonym transform | UNKNOWN | UNKNOWN | UNKNOWN | Requires dedicated audit |
| 12 | Word-order variation | UNKNOWN | — | — | — | — | Requires dedicated audit |
| 13 | Negation | UNKNOWN | — | — | — | — | Requires dedicated audit — clinically important (a negated symptom is a very different signal than an asserted one) |
| 14 | Temporal expressions | UNKNOWN | — | — | — | — | Requires dedicated audit |
| 15 | Severity modifiers | PARTIAL | Overlaps with the regex severity heuristic (FACT — heuristic exists, low-confidence by design per your own prior correction) | N/A — heuristic is post-hoc labeling, not training-time augmentation | N/A | Existing severity metrics | Distinct from a model-side robustness capability; do not conflate |
| 16 | Colloquial symptom descriptions | UNKNOWN | — | — | — | — | Requires dedicated audit |
| 17 | Mixed-script inputs | UNKNOWN | — | — | — | — | Requires dedicated audit |
| 18 | Rare/long-tail clinical terminology | UNKNOWN | — | — | — | — | Requires dedicated audit; interacts with NEISS-dominance risk (SPEC-04) |
| 19 | Hard-negative clinical presentations | PARTIAL | REPORTED hard-negative generation stage exists (`meditriage/multilingual/hard_negative/`, FACT per direct audit that the directory exists) | UNKNOWN | UNKNOWN | UNKNOWN | Requires dedicated audit |
| 20 | OOD inputs | PARTIAL | FACT — `data/ood_queries.csv` exists per direct audit | UNKNOWN | UNKNOWN | UNKNOWN | Requires dedicated audit |

**Binding paper-language constraint (DECIDED):** no claim in `docs/PAPER_RESULTS_DRAFT.md`, the dashboard, or any user-facing surface may say MediTriageAI "understands Hinglish" or similar unscoped language. Any capability claim must cite the specific dimension(s) demonstrated per this matrix, with the matrix's status column as the ceiling on how strongly the claim may be phrased.

**Required follow-on deliverable (not yet executed, UNKNOWN pending it):** a dedicated per-dimension audit filling in the UNKNOWN cells above — this is real work, not a formality, and should be scheduled before the paper's methodology section is finalized.

---

## 7. SPEC-05 — MODEL & ML SPECIFICATION (final authoritative model spec)

**Dual-head architecture (FACT):** shared transformer encoder → `Linear(hidden_size, 13)` specialist head + `Linear(hidden_size, 5)` severity head, both from the shared `[CLS]` representation.

**Backbone checkpoint identity — DECIDED, Option B:**

| Class name (unchanged) | Display name (unchanged for now) | **Canonical checkpoint identifier (mandatory everywhere else)** |
|---|---|---|
| `XLMRobertaLargeModel` | "XLM-RoBERTa-large" (class/display name preserved for compatibility) | `xlm-roberta-base` — **this exact string, not "large," must appear in every doc, model report, UI/model registry entry, and experiment artifact** |
| `IndicBertModel` | "IndicBERT" (class/display name preserved) | `google/muril-base-cased` — **this exact string, not "IndicBERT," must appear in every doc, model report, UI/model registry entry, and experiment artifact** |
| `MBertModel` | "mBERT" | `bert-base-multilingual-cased` (FACT, no discrepancy found) |
| `DistilBertMultilingualModel` | "DistilBERT-multi" | `distilbert-base-multilingual-cased` (FACT, no discrepancy found) |

**Prohibited descriptions (DECIDED, permanent until a CR changes this):** "XLM-RoBERTa-base → XLM-RoBERTa-large" and "MuRIL → IndicBERT" style substitutions are banned everywhere. A controlled class/display-name cleanup may occur later, during the `src/`→`meditriage/` migration, through normal change control — not silently, not now.

**Embedding initialization — CURRENT fact, frozen (DECIDED):** variant/OOV token embeddings are initialized via **single-anchor vector cloning** (`embedding_layer.weight.data[variant_id] = embedding_layer.weight.data[anchor_id].clone()`, `src/vocab_injection.py:56-68`, FACT). The paper draft's claim of multi-anchor mean initialization (`docs/PAPER_RESULTS_DRAFT.md:178-180`) is **CONTRADICTED** by this and must be corrected before submission. Multi-anchor mean initialization is recorded as **FUTURE-OPTIONAL** — a candidate ablation, not a retrofit into current results.

**E-PATH-CO-REASON — RESEARCH-EXPERIMENTAL (FACT status, unchanged):** DCCF/AMCO/DCES/DCRR/CTB/DCP modules fully implemented, 58 passing unit tests on interfaces, full training campaign not yet validated. **Binding rule (DECIDED, consistent with your "do not claim E-PATH superiority before DGX evidence" instruction):** E-PATH is reported in a separately-labeled subsection of any results table, never pooled with the four production backbones, until it has a complete, statistically-validated DGX training campaign under this same frozen spec.

**Joint loss (FACT):** masked Focal Loss, γ=2.0, α=1.0 (specialist), β=1.2 (severity), `ignore_index=-1`.

---

## 8. SPEC-06 — EVALUATION & STATISTICAL VERIFICATION

**Metrics inventory — FACT, confirmed by direct audit, broader than earlier drafts assumed:** Macro-F1, Weighted-F1, Top-1/2/3 accuracy, macro one-vs-rest AUROC, full per-class precision/recall/F1 (both tasks), severity MAE, ordinal confusion breakdown (exact-match / adjacent |Δ|=1 / **dangerous |Δ|≥2**), ECE, MCE, Brier score, NLL, Cohen's Kappa, McNemar's test, 1,000-resample bootstrap 95% CIs, robustness testing under perturbation.

### 9. Primary and secondary metrics — DECIDED, frozen before the DGX benchmark campaign

**Primary (drives "best performing model" ranking, FR-EVAL-02):**
- Specialist Macro-F1
- Severity Macro-F1

**Secondary (reported, informative, not the ranking criterion):**
- Weighted-F1
- Balanced Accuracy
- AUROC where statistically appropriate
- Calibration metrics (ECE/MCE/Brier/NLL)
- Severity MAE
- Dangerous severity confusion rate (|Δ|≥2)
- Multilingual robustness metrics (per Section 6's matrix, once populated)
- OOD metrics
- Safety/red-flag metrics (once the evaluation dataset in Section 10 exists)
- Practical/operational measurements (latency, resource requirements — see Section 8 of the practical-usefulness decision below)

**Binding rule (DECIDED, restated because it's the actual safety mechanism here):** this metric set is pre-registered now, before the benchmark campaign runs, and must not be changed retrospectively to favor a more flattering result. Any change requires a Change Request.

### Statistical methodology (unchanged from v0.3's correction, restated as final):
- **Macro-F1/Weighted-F1 comparisons:** paired bootstrap resampling of the metric difference; report absolute difference, 95% CI, whether it excludes zero, exact protocol, seed, resample count.
- **Accuracy/error-rate comparisons:** McNemar's test only, with full contingency table, statistic, p-value, exact-vs-asymptotic method, multiple-comparison correction.
- **Four non-conflatable labels:** Best performing model (primary metric); statistically supported improvement (survives the appropriate test above, corrected for multiple comparisons); most novel architecture (architectural-contribution judgment by you/Dr. Tripathi, not metric magnitude); most practically useful model (multidimensional report, Section 11 below — explicitly not a composite score at this stage).

**Safety-gatekeeper thresholds — DECIDED, TBD by default:** no arbitrary numerical threshold (e.g., a specific ECE cutoff) is frozen as normative. A threshold becomes normative only if supported by (A) an authoritative source (cited) or (B) an explicitly documented empirical methodology. Until then, every such threshold is marked **TBD / NOT CLINICALLY VALIDATED** in every report — the number is *reported*, never used as a pass/fail gate, until its basis is established.

---

## 10. SPEC-07 — CLINICAL SAFETY, ETHICS, GOVERNANCE

**Disclaimer (INTENT, non-negotiable):** present, byte-identical, everywhere (Section 1).

**Red-flag mechanism — DECIDED framing, permanent:** described everywhere as **"unaudited deterministic heuristic fallback."** Never described as a validated safety system. Current implementation (FACT): 8 hardcoded keywords in `scripts/serve_api.py` (`chest pain`, `radiation`, `loss of consciousness`, `severe bleeding`, `stroke`, `slurred speech`, `suicide`, `gunshot`), <0.60 confidence escalation threshold. No ground-truth evaluation dataset exists (FACT, confirmed by direct audit — `data/ood_queries.csv` and other data files contain only `patient_id`, `text`, `specialist_label`, `severity_label`, zero red-flag annotations).

**Red-flag evaluation dataset — DECIDED to build, size/strata TBD pending statistical rationale:**

Required strata (DECIDED framework, not yet populated):
- Genuine red flags
- Non-red-flag hard negatives
- Spelling variants
- Hinglish variants
- Paraphrases
- Synonyms
- Indirect descriptions
- Negation
- Temporal variation
- Long inputs (testing the 512-token truncation interaction specifically)
- Late-occurring red flags (near/after the truncation boundary)
- Symptom combinations

**Ground-truth adjudication methodology:** UNKNOWN — must be documented (who adjudicates, what counts as authoritative) before the dataset is built, not after.

**Size:** TBD — determined by an explicit statistical power/coverage rationale once the construction method and adjudication process are set, not by an arbitrary round number.

**FMEA table (carried forward, unchanged substance):**

| Failure mode | Likelihood | Severity | Mitigation |
|---|---|---|---|
| Red-flag keyword miss on paraphrased/Hinglish/indirect emergency symptom | UNKNOWN (untested) | Critical | Adversarial dataset above; never present as validated until it exists |
| Model under-triages a low-volume, high-acuity class due to NEISS dominance | Plausible per SPEC-04 | Critical | Per-class recall reporting |
| Truncation drops a late-disclosed symptom | Plausible, rate UNKNOWN | High | Truncation-rate metric; consider deterministic tail-window red-flag re-scan (cheap, doesn't require retraining) |
| Training artifact severely English-dominant despite multilingual pipeline existing | **UNKNOWN, unverified, conflicting reports** | Critical if true (invalidates any Hinglish claim) | DATASET-GATE-01 (Section 5) + the pending language-distribution audit (Section 5a) |

---

## 11. SPEC-08 — UX/UI & DESIGN SYSTEM (activated)

Screens: intake/demo (disclaimer above the fold), prediction result (calibration-aware confidence per FR-UX-01, red-flag panel with the mandatory phrase), model-zoo comparison dashboard (four separate, non-collapsed labels per Section 9), data-provenance view (renders directly from SPEC-04's tiering + text-provenance tables — single source of truth by import, not duplication), limitations/methodology page (first-class nav item: severity sparsity, synthetic share by A/B/C category, regex-circularity risk, NEISS dominance, truncation rate, red-flag recall status explicitly UNKNOWN, multilingual robustness matrix status).

**"Practical usefulness" display (per your Section 8 decision):** the dashboard renders this as a **multidimensional report**, not a single score — specialist routing quality, severity quality, dangerous error rate, calibration, multilingual robustness (per Section 6's matrix), OOD behavior, explainability (if `src/explainability.py` is ported), latency, resource requirements, failure modes. A composite utility score is explicitly **not built** at this stage; if wanted later, it requires its own proposal and approval.

Design baseline: WCAG 2.1 AA. Explicitly rejected: gamification, competitive leaderboard framing beyond the neutral comparison dashboard, conversational "chat with the triage AI" interface.

---

## 12. SPEC-09 — API & INTEGRATIONS (activated)

Endpoints: `POST /predict`, `GET /health`, `GET /version` (FR-OPS-01), `GET /metrics`, `GET /model-zoo`.

Response envelope (structural sketch, not yet a frozen contract at the field-name level): prediction (top-3 department + severity), confidence (model softmax + calibration reliability, kept distinct per FR-API-01), label-provenance context, red-flag status (with the mandatory phrase), disclaimer.

**Minimum security contract (per your Section 11 decision):** HTTP Basic Auth is acceptable for a trusted-network research demo under Track C's own scope definition; a token-based upgrade is a documented trigger-based decision point (SPEC-14), not something that blocks this freeze. Specific auth implementation may be finalized during implementation, provided it stays within this contract.

---

## 13. SPEC-10 — TESTING, CI/CD, DEFINITION OF DONE

Test matrix: unit (444-test baseline) → integration (dataset→model→eval round trip) → regression (SPEC-03 migration parity) → adversarial safety (Section 10's red-flag strata, once the dataset exists) → UX snapshot (disclaimer linter, FR-SAFETY-02) → deployment smoke test (SPEC-13).

CI/CD gates: no new TODO/placeholder strings; no test weakening without linked issue; disclaimer/red-flag-phrase diffs require explicit human approval; frozen-spec interface changes require a CR reference in the PR; any "port to `meditriage/`" PR requires an attached parity-test result.

---

## 14. SPEC-11 — TRACEABILITY MATRIX

| Product requirement (SPEC-01) | Software FR (SPEC-02) | Architecture/module | Verification | Status |
|---|---|---|---|---|
| Disclaimer everywhere | FR-SAFETY-01 | dashboard/API/CLI templates | Snapshot test | PROPOSED test |
| No training on unverified dataset | FR-DATA-03 (DATASET-GATE-01) | `meditriage/builder/`, training entrypoints | Gate report + checksum check | DECIDED, not yet implemented |
| Text provenance never pooled | FR-TEXT-01 | SPEC-04 schema | Schema field presence test | DECIDED, not yet implemented |
| Novelty paragraph auto-generated | FR-METRICS-01 | `src/metrics.py`/equivalent | No hand-authored claims anywhere | NOT YET IMPLEMENTED |
| Config hash real, not placeholder | FR-EVAL-03 | `src/evaluation.py`, `meditriage/training/config.py` | Placeholder string absent; hash matches config object | PENDING migration verification |
| Primary metrics pre-registered | FR-EVAL-02 | evaluation pipeline | Dated pre-registration artifact exists before benchmark run | DECIDED |
| Red-flag never overclaimed | FR-SAFETY-02 | all user-facing docs/UI | Doc-linter CI check | PROPOSED |
| Backbone checkpoint identity accurate | (new, unassigned FR — recommend **FR-MODEL-02**) | model reports, dashboard, UI/model registry, experiment artifacts | Grep/lint check for prohibited name substitutions | DECIDED policy, lint not yet built |
| `meditriage/` independence from `src/` | (tracked via SPEC-03 migration table) | Priority 1–4 module list | Parity test per module | MIGRATION in progress, Priority 1 not yet started |
| Multilingual claims scoped to matrix status | (new, recommend **FR-UX-03**) | dashboard limitations page, paper draft | Manual review against Section 6 matrix | DECIDED framework, per-dimension audit pending |

*(This is a representative, not line-exhaustive, matrix — sufficient to demonstrate the tracing mechanism works end-to-end; expanding it fully to every FR-ID is implementation-phase housekeeping, not a blocker to freeze.)*

---

## 15. SPEC-11b — RISK REGISTER (consolidated, all versions)

| # | Risk | Status | Mitigation |
|---|---|---|---|
| 1 | Regex-label circularity (specialty × severity heuristics both keyword-based) | FACT-grounded risk | Stratified reporting by provenance tier |
| 2 | Severity sparsity (1.3%) + 63% synthetic | FACT | Report synthetic share; sensitivity analysis excluding synthetic rows |
| 3 | NEISS dominance (70%, injury-only) | FACT | Per-class recall reporting |
| 4 | 512-token truncation drops late red flags | FACT (mechanism), rate UNKNOWN | Truncation-rate metric; tail-window red-flag re-scan (FUTURE-OPTIONAL, cheap) |
| 5 | `src/`↔`meditriage/` migration stalls or silently changes numbers | Structural risk | Hard gate (Section 3) + parity tests before any caller switch |
| 6 | Backbone naming implies larger/different models than actually used | CONTRADICTED, now resolved by policy | Option B naming policy (Section 7), prohibited-substitution lint |
| 7 | Paper claims multi-anchor mean embedding init; code does single-anchor clone | CONTRADICTED, now resolved | SPEC-05 canonical text corrected; paper draft correction required before submission |
| 8 | Red-flag mechanism has unknown, unaudited recall | FACT (no dataset exists) | Adversarial dataset (Section 10), strata defined, size/adjudication TBD |
| 9 | Prior training artifact may be severely English-dominant | **UNKNOWN, unverified, conflicting reports** | DATASET-GATE-01 (structural fix) + pending language-distribution audit (Section 5a) |
| 10 | "Synthetic" data pooled without distinguishing evidentiary strength (deterministic transform vs. templated vs. LLM) | FACT — ambiguity existed in prior spec drafts | Text-provenance taxonomy (Section 4), retroactive dataset audit required |
| 11 | "Hinglish support" claims risk overstating a controlled phonetic layer as general understanding | FACT — gap identified | Multilingual Robustness Audit Matrix (Section 6), binding paper-language constraint |
| 12 | E-PATH results could be presented alongside validated baselines, overstating its status | Structural risk, not yet materialized | Separate-subsection reporting rule (Section 7) |
| 13 | Arbitrary safety thresholds could be mistaken for clinical validation | Structural risk | TBD-by-default rule (Section 9) |

---

## 16. SPEC-12 — ARCHITECTURAL DECISION RECORDS (list, full status)

- **ADR-001:** `meditriage/`'s own dependencies on `src/` migrated in audited-risk order, not a blanket cutover — DECIDED, migration not started.
- **ADR-002:** Backbone naming/checkpoint policy — Option B (preserve class/display names, canonicalize checkpoint identifiers everywhere else, controlled cleanup later via CR) — DECIDED.
- **ADR-003:** Embedding initialization — single-anchor clone (current, frozen), multi-anchor mean (future ablation) — DECIDED, paper correction required.
- **ADR-004:** Red-flag evaluation methodology — "unaudited heuristic fallback" framing permanent (DECIDED); dataset construction/size/adjudication still PROPOSED, strata DECIDED.
- **ADR-005:** Severity synthetic-data policy (`fedmml_ed_triage` inclusion/weighting/reporting) — carried from backlog, not yet formally written.
- **ADR-006:** Evaluation/model-selection policy — primary/secondary metric split, four-label system, McNemar restricted to paired-accuracy, paired bootstrap for Macro-F1, no retrospective selection — DECIDED.
- **ADR-007:** Track C scope boundary (product-grade demo vs. clinical deployment) — DECIDED.
- **ADR-008 (new):** DATASET-GATE-01 as a binding pre-training requirement — DECIDED.
- **ADR-009 (new):** Text-provenance taxonomy (SOURCE/A/B/C) as a required field, orthogonal to label provenance — DECIDED.
- **ADR-010 (new):** 20-dimension Multilingual Robustness Framework adopted; per-dimension status audited, not assumed — DECIDED.
- **ADR-011 (new):** Practical usefulness reported as a multidimensional report, not a composite score, at this stage — DECIDED.
- **ADR-012 (new):** Safety-gatekeeper thresholds default to TBD absent an authoritative or documented-empirical basis — DECIDED.

*(Full ADR prose — context/decision/consequences/alternatives-considered — is an implementation-phase writing task; the decisions themselves are locked above.)*

---

## 17. SPEC-13 — DEPLOYMENT, OPERATIONS & OBSERVABILITY

Reproducible build (containerized or equivalent, pinned deps — already partially FACT via `requirements.txt`/`environment.yml`); `/version` endpoint (FR-OPS-01); request/error logging with no patient-identifiable content logged (ties to SPEC-14); dashboard uptime explicitly not SLA-bound (research demo, not monitored production service — stated plainly to avoid implying an operational guarantee that doesn't exist).

**Hosting target — NON-BLOCKING per your decision.** The deployment *contract* is specified above regardless of where it eventually runs; the specific target (local-only, university server, cloud) is an implementation-time decision that doesn't block this freeze.

**DGX artifact tagging (ties to Section 3 and DATASET-GATE-01 item 18):** every DGX-produced training/evaluation/benchmark artifact carries both the frozen spec version and the dataset checksum it was executed against.

---

## 18. SPEC-14 — SECURITY & PRIVACY

FR-SEC-01 (no PHI ingestion path, DECIDED permanent constraint) is the anchor requirement. Auth: HTTP Basic acceptable now (Section 12); token-based upgrade is a documented, non-blocking trigger point. Dependency supply chain: existing pinning extended to a periodic review cadence tied to spec-version bumps. Logging hygiene: no verbatim patient-presentation text logged in a way that would recreate a PHI-adjacent record, even though source datasets are public — a conservative default given the clinical framing.

---

## 19. SPEC-15 — CHANGE CONTROL & RELEASE MANAGEMENT

CR mandatory fields: motivation, affected SPEC/FR/ADR IDs, test delta, approval reference. Versioning: `MAJOR.MINOR.PATCH` per document within the single consolidated baseline; a release is a named, tagged spec version + the code commit it corresponds to. No autonomous agent may edit a frozen requirement — only propose a CR.

---

## 20. DGX EXECUTION CONTRACT

DGX executes exclusively against a version-pinned frozen spec (once frozen). Before any DGX work begins on this project:

1. Verify the local copy of the frozen spec's checksum against its manifest (once Section E's frozen-folder/manifest procedure from the prior round is executed — not yet, since nothing is frozen).
2. Run **DATASET-GATE-01** end-to-end (Section 5) before any training campaign; produce and file the language-distribution audit report (Section 5a) as the first concrete action, since it's cheap, resolves a live UNKNOWN, and gates everything downstream.
3. If an artifact fails the gate, classify it NOT ELIGIBLE — no undocumented rescue.
4. Tag every output artifact with spec version + dataset checksum.
5. If the spec is found ambiguous, contradictory, or infeasible during DGX execution, stop and file a Change Request — do not resolve by assumption, and do not silently adjust the spec to fit what DGX discovers.
6. Long-running/full test campaigns, multi-GPU execution, final benchmark generation, and reproducibility runs belong here; day-to-day implementation stays LOCAL.

---

## 21. GSD OPERATING CONTRACT

Tasks scoped to a single SPEC section or FR-ID, never "implement the spec" as one task. `TREE IMPACT` declaration required before code, citing the FR/SPEC-ID being implemented. Ambiguity is surfaced to you, not resolved by assumption. Migration work follows the Priority 1→4 order from Section 3, with parity tests required before any caller switch.

## 22. RALPH OPERATING CONTRACT

Scoped to a single, atomic, testable requirement per run. Never touches red-flag keyword lists, the disclaimer string, backbone naming, or any other DECIDED item without an approved CR referenced in its task input. Output always accompanied by actual test-run pass/fail counts, never self-reported "done."

## 23. CODERABBIT GOVERNANCE GATES

Block merge if: new TODO/placeholder string introduced; test weakened/skipped without linked issue; disclaimer or red-flag mandatory phrase modified; frozen-spec interface changed without a CR reference in the PR; a `src/`→`meditriage/` port PR lacks an attached parity-test result; a backbone naming change violates the Option B prohibited-substitution rule (Section 7); any PR introduces a hardcoded safety/calibration threshold without a cited authoritative source or documented empirical methodology (Section 9).

---

## 24. FINAL PRE-FREEZE CHECKLIST

This is the last gate before you authorize `v1.0.0-FROZEN`. Items remaining, in priority order:

1. **Run the language-distribution audit (Section 5a) on DGX.** This is the single highest-value remaining action — cheap, resolves a live factual UNKNOWN, and its result (PASS/FAIL against DATASET-GATE-01) materially affects what can honestly be claimed about the existing multilingual pipeline in the frozen spec's own risk register (Section 15, risk #9).
2. **Confirm `config.get_hash()`'s existence/determinism** (or authorize its implementation) as part of the Priority-2 migration work (Section 3) — needed to actually close FR-EVAL-03, not just specify it.
3. **Begin the Priority-1 (CRITICAL) migration parity work** (Section 3) — this is the true hard gate before any frozen-spec paper results run, and it hasn't started yet.
4. **Schedule the per-dimension Multilingual Robustness audit** (Section 6) to fill in the UNKNOWN cells — needed before the paper's methodology section can honestly describe multilingual capability.
5. **Decide the red-flag dataset construction method** (Section 10) — strata are DECIDED, but source/adjudication/size are not, and nothing in that dataset can be built until they are.
6. Confirm you're satisfied that hosting target and auth-upgrade trigger remain appropriately non-blocking (Sections 17/18) — no action needed unless you disagree.

**None of items 1–5 need to complete before you freeze this specification** — the spec itself can freeze now, describing target state, migration plans, and gates precisely, with items 1–5 tracked as the first wave of post-freeze implementation work under the GSD/Ralph/DGX contracts above. Freezing the spec is not the same as claiming the work is done; it's committing to *what* "done" means so implementation can proceed without re-litigating requirements. That said, item 1 is cheap enough that you may prefer to have its result in hand before freezing, purely because it could change how strongly Section 15's risk #9 gets worded in the frozen document — your call.

**I am stopping here, as instructed.** No frozen folder created, no checksum manifest generated, no `v1.0.0-FROZEN` label applied anywhere in this document or elsewhere. Awaiting your explicit authorization to proceed to the actual freeze procedure (Sections D–F from the prior round: write to `/specs/frozen/v1.0.0/`, generate `MANIFEST.json`, update `VERSION`/`.planning/STATE.md`).