# MediTriageAI Specification Baseline v1.0.0
### Status: **v1.0.0-FROZEN**
### Authoritative specification baseline for MediTriageAI_Data_Engine

---

## 0. HOW TO READ THIS DOCUMENT

This is the single, physically-consolidated, self-contained specification for MediTriageAI. It incorporates the Final Candidate v1.0.0 and all approved Amendments (1–11). No external reference (conversation, prior draft, or commentary) is authoritative — this document is the contract.

### Status Tags

Applied per-statement, never assumed:

- **FACT** — verified directly against repository evidence (file/line citations from direct Antigravity tool-trace audits).
- **IMPLEMENTED** — exists in the repository today, verified.
- **PARTIAL** — exists in some form but doesn't meet the full acceptance criterion yet.
- **MISSING** — does not exist yet; a target and acceptance criterion are specified regardless.
- **UNKNOWN** — existence/correctness not yet verified; must be checked, not assumed either way.
- **MIGRATION** — mid-transition from a `src/` implementation to a `meditriage/` implementation.
- **RESEARCH-EXPERIMENTAL** — implemented but not validated to production standard (E-PATH is the standing example).
- **TBD** — deliberately left unset because no defensible basis exists yet to set it (used for thresholds, dataset sizes — never filled with a plausible-sounding placeholder).
- **INTENT** — stated research/product north star.
- **PROPOSED** — a recommendation still open for approval.
- **DECIDED** — locked by explicit approval.
- **CONTRADICTED** — a documentation/paper claim disproven by code evidence.
- **FUTURE-OPTIONAL** — a candidate enhancement, not part of current baseline.

### Five-Way System-State Classification

Applied throughout: **CURRENT** / **TARGET** / **MIGRATION** / **RESEARCH-EXPERIMENTAL** / **FUTURE-OPTIONAL**.

### Governing Rule (Amendment 2)

**Freezing v1.0.0 means freezing the requirements, architecture, governance model, acceptance criteria, and change-control contract. It does NOT mean claiming any requirement is already implemented.** Every requirement carries one of the implementation-status values listed above; freezing the spec locks *what the requirement is*, not *whether it's done*. This rule prevents "we froze the spec" from ever being misread as "the system is done."

### Terminology Constraint (Amendment 11)

**"Frozen" describes only this approved specification baseline** (`docs/specification/frozen/v1.0.0/`) unless explicitly qualified otherwise. It must never be used to describe currently implemented code, the current dataset, current model weights, or current benchmark results without an explicit qualifier (e.g., a specific checksum). Unqualified "frozen" always means this specification document, nothing else.

**Nothing is promoted to FACT without repository evidence in this document.** Where prior commentary-style input hasn't been independently verified, it remains explicitly UNKNOWN.

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
| FR-DATA-01 | Every label carries source-type provenance (`DIRECT`/`MAPPED`/`INFERRED`) | Canonical schema row always has `label_provenance`; test asserts no row missing it | IMPLEMENTED (FACT) |
| FR-DATA-02 | EXCLUDED datasets never enter active training without explicit override | Test: loading `l3cube_code_mixed`/`medical_meadow_medqa`/`medqa_usmle` raises by default | UNKNOWN — not confirmed enforced; add as MIGRATION-phase test |
| FR-DATA-03 | DATASET-GATE-01 — see full spec, Section 5 | Gate report generated and passing before any training run | DECIDED, binding — MISSING |
| FR-TEXT-01 | Every row's text (not just label) carries a text-provenance category: `SOURCE`, `A` (deterministic linguistic augmentation), `B` (rule-based/templated construction), `C` (LLM-generated) | Field present, queryable, never pooled into an anonymous "synthetic" bucket | DECIDED — MISSING |
| FR-MODEL-01 | Model zoo supports ≥2 backbones through one eval harness | `pytest tests/test_model_zoo.py` | IMPLEMENTED (FACT) |
| FR-MODEL-02 | All reports, registries, experiment artifacts, model metadata, documentation, and UI descriptions use exact pretrained checkpoint identifiers, not misleading class/display names | Grep/lint check for prohibited name substitutions | DECIDED — MISSING |
| FR-METRICS-01 | Novelty paragraph generated programmatically from evaluation JSON | `src/metrics.py`/equivalent exists; no file contains a hand-authored novelty claim | MISSING — critical-path build item |
| FR-EVAL-01 | Evaluation config hash computed from actual config, never a placeholder | `"TODO_config_hash"` at `src/evaluation.py:137` removed; replaced by deterministic hash | CURRENT = broken (FACT); TARGET: Gate 3 |
| FR-EVAL-02 | Primary model-selection metrics frozen before the final DGX benchmark campaign; no retrospective metric selection | Metric names appear in a version-controlled pre-registration artifact dated before the benchmark run | DECIDED — MISSING (policy locked, artifact not yet created) |
| FR-EVAL-03 | Deterministic, canonical configuration hashing mechanism | If `config.get_hash()` (or equivalent) exists and is deterministic/canonical, use it; if not, implement one as part of the migration | DECIDED — UNKNOWN (existence unverified; resolve during Gate 3) |
| FR-SAFETY-01 | Disclaimer present, byte-identical, across CLI/API/dashboard | Snapshot test asserts presence and exact match | MISSING |
| FR-SAFETY-02 | Documentation linter fails build if red-flag mentions lack the mandatory "unaudited deterministic heuristic fallback" phrase nearby | Grep-based CI check | MISSING |
| FR-UX-01 | Prediction surfaces show calibration-aware confidence, not bare softmax | UI value sourced from actual ECE/reliability data, not hardcoded | MISSING |
| FR-UX-02 | Disclaimer is structurally required in response schema/templates, not optional copy | Removing the disclaimer field from any template/response schema fails a snapshot test | MISSING |
| FR-UX-03 | Multilingual capability claims scoped to Robustness Matrix status | Manual review at each release against Section 6 matrix; no claim exceeds what the status column supports | PARTIAL — rule binding, enforcement mechanism not yet built |
| FR-API-01 | API separates provenance-derived confidence from model confidence | Distinct response fields; test asserts no conflation | MISSING |
| FR-OPS-01 | `/version` (or equivalent) reports exact model checkpoint hash + spec version | Test hits endpoint, asserts match to build manifest | MISSING |
| FR-SEC-01 | No PHI ingestion path anywhere in the system | Architecture review checklist item; no code path accepts identifiable patient data by design | IMPLEMENTED by design (FACT) |

**Definition of Done (hosted formally in SPEC-10):** code merged + test passing + (if touching labels/severity/red-flags/multilingual claims) limitations doc updated in the same PR + no new TODO/placeholder strings introduced.

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

**Execution environment boundary (DECIDED, binding governance):**
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

**Label-quality risk table:**

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
11. **Language-distribution report generated** (the direct structural fix for the English-dominance risk).
12. Class-distribution report generated.
13. Train/validation/test isolation explicitly verified (not merely assumed from the leakage-safe split design).
14. Provenance recorded for every generated sample, using the SPEC-04 text-provenance taxonomy (`SOURCE`/`A`/`B`/`C`).
15. Synthetic/generated-vs-source proportions reported, broken out by taxonomy category, never pooled.
16. Final dataset receives a SHA-256 checksum.
17. Training configuration references that exact checksum.
18. Any DGX training run records the dataset checksum in its output artifacts.

**Binding failure behavior (DECIDED):** the training entrypoint must **fail loudly** — not silently fall back — if the checksum it's configured with doesn't match the dataset actually present on disk. The system must never silently train on an older builder output when a newer canonical dataset has been specified.

**Acceptance criterion for item 11:** the language-distribution report must be generated and reviewed **before** any DGX training run is authorized, not as a post-hoc audit artifact.

### 5a. Historical language-distribution audit — DECIDED, execution status: PENDING (Amendment 1)

The historical language-distribution audit remains **UNKNOWN/PENDING** at freeze time. It is **not required before specification freeze.** It becomes **GATE 1** (see Section 23) — the first post-freeze execution task — and **must run before any new official training campaign or before any reliance on previous multilingual benchmark claims.**

If the historical artifact fails DATASET-GATE-01, it is classified **NOT ELIGIBLE** for final benchmark use, per the binding rule already established. This does not change anything substantive — it only removes ambiguity about whether the audit blocks freeze. It does not.

**Required audit output format:**
```
artifact_path: <exact path>
artifact_checksum: <SHA-256>
row_count: <int>
language_counts: { en: <int>, hi_latn: <int>, hi_en_mixed: <int>, ... }
language_percentages: { ... }
specialist_distribution: { CARDIO_PULM: <pct>, ED: <pct>, ... (all 13) }
severity_distribution: { S1: <pct>, ..., S5: <pct> }
provenance_distribution: { SOURCE: <pct>, A: <pct>, B: <pct>, C: <pct> }
dataset_gate_status: PASS | FAIL, with the specific failing item numbers listed if FAIL
```

**Binding rule (DECIDED):** the reported ~99.42%-English figure from prior commentary-style input is assumed **neither true nor false** — it is determined from the actual artifact, once run. If an old training artifact fails DATASET-GATE-01, it is classified **NOT ELIGIBLE** for the final benchmark campaign — full stop, no undocumented rescue attempt, no partial-credit reasoning.

**This audit has not been run.** Its results are UNKNOWN until executed.

---

## 6. MULTILINGUAL ROBUSTNESS FRAMEWORK + AUDIT MATRIX (DECIDED framework, PER-DIMENSION STATUS MOSTLY UNKNOWN)

**Framework adopted, with the explicit caveat that adoption of the framework is not adoption of the claim that all 20 capabilities exist.** For each dimension, the required evidentiary chain is: **generation/source mechanism → training exposure → validation coverage → evaluation metric → acceptance criterion.** A dimension only counts as demonstrated once all five links are shown with evidence — not inferred from the pipeline diagram existing.

| # | Dimension | Status | Generation/Source | Training Exposure | Validation | Evaluation Metric | Acceptance Criterion |
|---|---|---|---|---|---|---|---|
| 1 | Standard English | IMPLEMENTED | SOURCE data (FACT — majority of corpora) | FACT | Standard test split | Existing Macro-F1 | Primary metric (Section 9) |
| 2 | Standard Hindi (Devanagari) | MISSING | Prior scope included Devanagari Hindi but current active build focuses on English + Hinglish (romanized) | N/A | N/A | N/A | Explicitly out of current scope — do not claim |
| 3 | Romanized Hindi | PARTIAL | REPORTED deterministic phonetic engine (`src/hinglish_perturbation.py`, FACT that the module exists per direct audit) | UNKNOWN — coverage in actual training runs not confirmed | UNKNOWN | UNKNOWN | Requires dedicated audit |
| 4 | English-Hindi code mixing | PARTIAL | Multilingual expansion pipeline stage referenced in project docs (REPORTED) | UNKNOWN | UNKNOWN | UNKNOWN | Requires dedicated audit |
| 5 | Phonetic transliteration | PARTIAL | Same variant-table mechanism as #3 | UNKNOWN | UNKNOWN | UNKNOWN | Requires dedicated audit |
| 6 | Common spelling variation | UNKNOWN | Not independently verified | UNKNOWN | UNKNOWN | UNKNOWN | Requires dedicated audit |
| 7 | Informal chat spelling | UNKNOWN | — | — | — | — | Requires dedicated audit |
| 8 | Abbreviations | UNKNOWN | — | — | — | — | Requires dedicated audit |
| 9 | Clinical shorthand | UNKNOWN | — | — | — | — | Requires dedicated audit |
| 10 | ASR-like transcription noise | PARTIAL | REPORTED "ASR noise" transform in the 10-style linguistic variation engine | UNKNOWN | UNKNOWN | UNKNOWN | Requires dedicated audit |
| 11 | Synonym variation | PARTIAL | REPORTED synonym transform | UNKNOWN | UNKNOWN | UNKNOWN | Requires dedicated audit |
| 12 | Word-order variation | UNKNOWN | — | — | — | — | Requires dedicated audit |
| 13 | Negation | UNKNOWN | — | — | — | — | Requires dedicated audit — clinically important |
| 14 | Temporal expressions | UNKNOWN | — | — | — | — | Requires dedicated audit |
| 15 | Severity modifiers | PARTIAL | Overlaps with the regex severity heuristic (FACT — heuristic exists) | N/A — heuristic is post-hoc labeling, not training-time augmentation | N/A | Existing severity metrics | Distinct from a model-side robustness capability; do not conflate |
| 16 | Colloquial symptom descriptions | UNKNOWN | — | — | — | — | Requires dedicated audit |
| 17 | Mixed-script inputs | UNKNOWN | — | — | — | — | Requires dedicated audit |
| 18 | Rare/long-tail clinical terminology | UNKNOWN | — | — | — | — | Requires dedicated audit; interacts with NEISS-dominance risk (SPEC-04) |
| 19 | Hard-negative clinical presentations | PARTIAL | REPORTED hard-negative generation stage exists (`meditriage/multilingual/hard_negative/`, FACT per direct audit that the directory exists) | UNKNOWN | UNKNOWN | UNKNOWN | Requires dedicated audit |
| 20 | OOD inputs | PARTIAL | FACT — `data/ood_queries.csv` exists per direct audit | UNKNOWN | UNKNOWN | UNKNOWN | Requires dedicated audit |

**Binding paper-language constraint (DECIDED):** no claim in `docs/PAPER_RESULTS_DRAFT.md`, the dashboard, or any user-facing surface may say MediTriageAI "understands Hinglish" or similar unscoped language. Any capability claim must cite the specific dimension(s) demonstrated per this matrix, with the matrix's status column as the ceiling on how strongly the claim may be phrased.

**Required follow-on deliverable (not yet executed, UNKNOWN pending it):** a dedicated per-dimension audit filling in the UNKNOWN cells above — this is real work, not a formality, and should be scheduled before the paper's methodology section is finalized.

---

## 7. SPEC-05 — MODEL & ML SPECIFICATION

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

**E-PATH-CO-REASON — RESEARCH-EXPERIMENTAL (FACT status, unchanged):** DCCF/AMCO/DCES/DCRR/CTB/DCP modules fully implemented, 58 passing unit tests on interfaces, full training campaign not yet validated. **Binding rule (DECIDED):** E-PATH is reported in a separately-labeled subsection of any results table, never pooled with the four production backbones, until it has a complete, statistically-validated DGX training campaign under this same frozen spec.

**Joint loss (FACT):** masked Focal Loss, γ=2.0, α=1.0 (specialist), β=1.2 (severity), `ignore_index=-1`.

---

## 8. SPEC-06 — EVALUATION & STATISTICAL VERIFICATION

**Metrics inventory — FACT, confirmed by direct audit:** Macro-F1, Weighted-F1, Top-1/2/3 accuracy, macro one-vs-rest AUROC, full per-class precision/recall/F1 (both tasks), severity MAE, ordinal confusion breakdown (exact-match / adjacent |Δ|=1 / **dangerous |Δ|≥2**), ECE, MCE, Brier score, NLL, Cohen's Kappa, McNemar's test, 1,000-resample bootstrap 95% CIs, robustness testing under perturbation.

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
- Practical/operational measurements (latency, resource requirements)

**Binding rule (DECIDED):** this metric set is pre-registered now, before the benchmark campaign runs, and must not be changed retrospectively to favor a more flattering result. Any change requires a Change Request.

### Statistical methodology (final):
- **Macro-F1/Weighted-F1 comparisons:** paired bootstrap resampling of the metric difference; report absolute difference, 95% CI, whether it excludes zero, exact protocol, seed, resample count.
- **Accuracy/error-rate comparisons:** McNemar's test only, with full contingency table, statistic, p-value, exact-vs-asymptotic method, multiple-comparison correction.
- **Four non-conflatable labels:** Best performing model (primary metric); statistically supported improvement (survives the appropriate test above, corrected for multiple comparisons); most novel architecture (architectural-contribution judgment, not metric magnitude); most practically useful model (multidimensional report — explicitly not a composite score at this stage).

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

**FMEA table:**

| Failure mode | Likelihood | Severity | Mitigation |
|---|---|---|---|
| Red-flag keyword miss on paraphrased/Hinglish/indirect emergency symptom | UNKNOWN (untested) | Critical | Adversarial dataset above; never present as validated until it exists |
| Model under-triages a low-volume, high-acuity class due to NEISS dominance | Plausible per SPEC-04 | Critical | Per-class recall reporting |
| Truncation drops a late-disclosed symptom | Plausible, rate UNKNOWN | High | Truncation-rate metric; consider deterministic tail-window red-flag re-scan (FUTURE-OPTIONAL, cheap) |
| Training artifact severely English-dominant despite multilingual pipeline existing | **UNKNOWN, unverified, conflicting reports** | Critical if true (invalidates any Hinglish claim) | DATASET-GATE-01 (Section 5) + the pending language-distribution audit (Section 5a) |

---

## 11. SPEC-08 — UX/UI & DESIGN SYSTEM (activated)

Screens: intake/demo (disclaimer above the fold), prediction result (calibration-aware confidence per FR-UX-01, red-flag panel with the mandatory phrase), model-zoo comparison dashboard (four separate, non-collapsed labels per Section 9), data-provenance view (renders directly from SPEC-04's tiering + text-provenance tables — single source of truth by import, not duplication), limitations/methodology page (first-class nav item: severity sparsity, synthetic share by A/B/C category, regex-circularity risk, NEISS dominance, truncation rate, red-flag recall status explicitly UNKNOWN, multilingual robustness matrix status).

**"Practical usefulness" display:** the dashboard renders this as a **multidimensional report**, not a single score — specialist routing quality, severity quality, dangerous error rate, calibration, multilingual robustness (per Section 6's matrix), OOD behavior, explainability (if `src/explainability.py` is ported), latency, resource requirements, failure modes. A composite utility score is explicitly **not built** at this stage; if wanted later, it requires its own proposal and approval.

Design baseline: WCAG 2.1 AA. Explicitly rejected: gamification, competitive leaderboard framing beyond the neutral comparison dashboard, conversational "chat with the triage AI" interface.

---

## 12. SPEC-09 — API & INTEGRATIONS (activated)

Endpoints: `POST /predict`, `GET /health`, `GET /version` (FR-OPS-01), `GET /metrics`, `GET /model-zoo`.

Response envelope (structural sketch, not yet a frozen contract at the field-name level): prediction (top-3 department + severity), confidence (model softmax + calibration reliability, kept distinct per FR-API-01), label-provenance context, red-flag status (with the mandatory phrase), disclaimer.

**Minimum security contract:** HTTP Basic Auth is acceptable for a trusted-network research demo under Track C's own scope definition; a token-based upgrade is a documented trigger-based decision point (SPEC-14), not something that blocks this freeze. Specific auth implementation may be finalized during implementation, provided it stays within this contract.

---

## 13. SPEC-10 — TESTING, CI/CD, DEFINITION OF DONE

Test matrix: unit (444-test baseline) → integration (dataset→model→eval round trip) → regression (SPEC-03 migration parity) → adversarial safety (Section 10's red-flag strata, once the dataset exists) → UX snapshot (disclaimer linter, FR-SAFETY-02) → deployment smoke test (SPEC-13).

CI/CD gates: no new TODO/placeholder strings; no test weakening without linked issue; disclaimer/red-flag-phrase diffs require explicit human approval; frozen-spec interface changes require a CR reference in the PR; any "port to `meditriage/`" PR requires an attached parity-test result.

---

## 14. SPEC-11 — TRACEABILITY MATRIX

The complete, normative FR-ID traceability matrix is maintained in `TRACEABILITY.md` within this frozen specification package. Every FR-ID is mapped to: SPEC section → implementation location/target → verification method → acceptance criterion → current status. No FR-ID is left unmapped and no status is left implicit.

See `TRACEABILITY.md` for the full matrix (18 FR-IDs: FR-DATA-01 through FR-DATA-03, FR-TEXT-01, FR-MODEL-01 through FR-MODEL-02, FR-METRICS-01, FR-EVAL-01 through FR-EVAL-03, FR-SAFETY-01 through FR-SAFETY-02, FR-UX-01 through FR-UX-03, FR-API-01, FR-OPS-01, FR-SEC-01).

---

## 15. SPEC-11b — RISK REGISTER

The complete risk register is maintained in `RISK_REGISTER.md` within this frozen specification package. All 13 identified risks with their current status and required mitigations are documented there.

---

## 16. SPEC-12 — ARCHITECTURAL DECISION RECORDS

ADR-001 through ADR-012 are maintained in the `ADRs/` directory within this frozen specification package. Each contains: Context, Decision, Consequences, and Alternatives Considered. The decisions themselves are listed here for reference:

- **ADR-001:** `meditriage/`'s dependencies on `src/` migrated in audited-risk order, not a blanket cutover — DECIDED.
- **ADR-002:** Backbone naming/checkpoint policy — Option B — DECIDED.
- **ADR-003:** Embedding initialization — single-anchor clone (current, frozen), multi-anchor mean (future ablation) — DECIDED.
- **ADR-004:** Red-flag evaluation methodology — "unaudited heuristic fallback" framing permanent — DECIDED.
- **ADR-005:** Severity synthetic-data policy — DECIDED.
- **ADR-006:** Evaluation/model-selection policy — DECIDED.
- **ADR-007:** Track C scope boundary — DECIDED.
- **ADR-008:** DATASET-GATE-01 as a binding pre-training requirement — DECIDED.
- **ADR-009:** Text-provenance taxonomy (SOURCE/A/B/C) — DECIDED.
- **ADR-010:** 20-dimension Multilingual Robustness Framework — DECIDED.
- **ADR-011:** Practical usefulness as multidimensional report, not composite score — DECIDED.
- **ADR-012:** Safety-gatekeeper thresholds default to TBD — DECIDED.

---

## 17. SPEC-13 — DEPLOYMENT, OPERATIONS & OBSERVABILITY

Reproducible build (containerized or equivalent, pinned deps — already partially FACT via `requirements.txt`/`environment.yml`); `/version` endpoint (FR-OPS-01); request/error logging with no patient-identifiable content logged (ties to SPEC-14); dashboard uptime explicitly not SLA-bound (research demo, not monitored production service).

**Hosting target — NON-BLOCKING.** The deployment *contract* is specified above regardless of where it eventually runs; the specific target is an implementation-time decision.

**DGX artifact tagging:** every DGX-produced training/evaluation/benchmark artifact carries both the frozen spec version and the dataset checksum it was executed against.

---

## 18. SPEC-14 — SECURITY & PRIVACY

FR-SEC-01 (no PHI ingestion path, DECIDED permanent constraint) is the anchor requirement. Auth: HTTP Basic acceptable now (Section 12); token-based upgrade is a documented, non-blocking trigger point. Dependency supply chain: existing pinning extended to a periodic review cadence tied to spec-version bumps. Logging hygiene: no verbatim patient-presentation text logged in a way that would recreate a PHI-adjacent record, even though source datasets are public.

---

## 19. SPEC-15 — CHANGE CONTROL & RELEASE MANAGEMENT

CR mandatory fields: motivation, affected SPEC/FR/ADR IDs, test delta, approval reference. Versioning: `MAJOR.MINOR.PATCH`; a release is a named, tagged spec version + the code commit it corresponds to. No autonomous agent may edit a frozen requirement — only propose a CR.

---

## 20. DGX EXECUTION CONTRACT

DGX executes exclusively against a version-pinned frozen spec. The exact ordered sequence (Amendment 8), which the DGX runner must follow with no steps skipped or reordered:

```
1. FROZEN SPEC CHECKSUM       — verify local frozen copy against MANIFEST.json (Gate 0)
2. DATASET AUDIT               — language-distribution + full DATASET-GATE-01 report (Gates 1–2)
3. DATASET-GATE-01             — pass/fail determination; halt here if FAIL
4. TRAINING CONFIG VALIDATION  — config hash computed and matched against declared value (Gate 3 dependency)
5. TRAINING                    — executed only if steps 1–4 all passed (Gate 8)
6. EVALUATION                  — full metric suite per SPEC-06 (Gate 9)
7. STATISTICAL VERIFICATION    — paired bootstrap / McNemar per the approved methodology (Gate 9)
8. ARTIFACT MANIFEST           — every output tagged with spec version + dataset checksum (SPEC-13)
9. REPRODUCIBILITY CHECK       — re-run verification confirming the artifact manifest's claims are reproducible
```

Any failure at steps 1–4 halts the sequence before compute is spent on steps 5–9.

If the spec is found ambiguous, contradictory, or infeasible during DGX execution, stop and file a Change Request — do not resolve by assumption.

---

## 21. GSD OPERATING CONTRACT

Tasks scoped to a single SPEC section or FR-ID, never "implement the spec" as one task. `TREE IMPACT` declaration required before code, citing the FR/SPEC-ID being implemented. Ambiguity is surfaced, not resolved by assumption. Migration work follows the Priority 1→4 order from Section 3, with parity tests required before any caller switch.

---

## 22. RALPH OPERATING CONTRACT

Scoped to a single, atomic, testable requirement per run (Amendment 6). Never touches red-flag keyword lists, the disclaimer string, backbone naming, or any other DECIDED item without an approved CR referenced in its task input. Output always accompanied by actual test-run pass/fail counts, never self-reported "done."

**Ralph is not permitted to autonomously execute across multiple gates in one run.** Every Ralph invocation must be scoped with:

```
objective: <one bounded, atomic task>
fr_or_spec_reference: <exact FR-ID or SPEC section>
gate: <which gate from Section 23 this belongs to>
allowed_files: [<explicit file/directory allowlist>]
verification_command: <exact command to run>
stop_condition: <what makes this run done or blocked>
expected_acceptance_criterion: <copied verbatim from the traceability matrix>
```

A Ralph run that touches files outside its declared allowlist, or whose stated objective spans more than one gate, is out of contract regardless of whether its output looks correct.

---

## 23. CODERABBIT GOVERNANCE GATES

Block merge if: new TODO/placeholder string introduced; test weakened/skipped without linked issue; disclaimer or red-flag mandatory phrase modified; frozen-spec interface changed without a CR reference in the PR; a `src/`→`meditriage/` port PR lacks an attached parity-test result; a backbone naming change violates the Option B prohibited-substitution rule (Section 7); any PR introduces a hardcoded safety/calibration threshold without a cited authoritative source or documented empirical methodology (Section 9).

**CodeRabbit's "approve" result is not equivalent to passing the specification's acceptance criteria (Amendment 7).** CodeRabbit performs code-quality/style/security-pattern review; the traceability matrix's verification method remains the sole authority on whether a requirement is satisfied.

---

## 24. POST-FREEZE IMPLEMENTATION GATES (Amendment 5)

These gates are part of the frozen governance contract. GSD/Ralph must not skip a gate or begin a later gate before an earlier one's exit criterion is met, except where explicitly marked parallelizable.

| Gate | Name | Entry criterion | Exit criterion | Primary FR-IDs / SPEC sections |
|---|---|---|---|---|
| **0** | Specification integrity | Freeze authorized | Frozen file checksums (MANIFEST.json) verified against on-disk copy before any other gate proceeds | Freeze package itself |
| **1** | Historical dataset language audit | Gate 0 passed | Language-distribution audit report (Section 5a format) produced and filed; historical artifact classified ELIGIBLE or NOT ELIGIBLE against DATASET-GATE-01 | FR-DATA-03 (partial), Section 5a |
| **2** | Dataset Gate implementation | Gate 0 passed (parallelizable with Gate 1) | DATASET-GATE-01 mechanism itself exists in code: gate-report generator, checksum verification, fail-loudly training entrypoint, text-provenance field wired through | FR-DATA-03, FR-TEXT-01 |
| **3** | Priority-1 `src→meditriage` parity | Gate 0 passed (parallelizable with Gates 1–2) | `src/model.py`, `src/trainer.py`, `src/dataset.py`, `src/schema.py` ported with verified numerical/behavioral parity; `config.get_hash()` existence resolved (implemented if missing) | FR-EVAL-01, FR-EVAL-03, SPEC-03 Priority-1 table |
| **4** | Multilingual robustness audit | Gate 0 passed (parallelizable) | All 20 dimensions in the Robustness Matrix (Section 6) have a determined status (not UNKNOWN) with evidence for each of the five required links | FR-UX-03 |
| **5** | Red-flag evaluation dataset | Gate 0 passed (parallelizable) | Construction method, adjudication methodology, and statistically-justified size are all determined and documented; dataset built | SPEC-07 strata table |
| **6** | Production implementation | Gates 1–5 substantially complete for the modules each touches | All remaining MISSING FR-IDs in the traceability matrix implemented | Full traceability matrix |
| **7** | Full test suite | Gate 6 complete | 444+ existing tests plus all new safety/adversarial/snapshot/lint tests pass | SPEC-10 |
| **8** | DGX canonical training | Gates 2, 3, 7 passed; Gate 1 result reviewed | Training executed under a DATASET-GATE-01-passed dataset, against pre-registered primary metrics (FR-EVAL-02 artifact filed *before* this gate opens) | FR-EVAL-02, DGX Execution Contract |
| **9** | DGX evaluation / statistical verification | Gate 8 complete | Full evaluation run: paired bootstrap for Macro-F1 comparisons, McNemar for accuracy comparisons, calibration/safety/multilingual/OOD metrics all reported | SPEC-06 methodology |
| **10** | Final research artifact generation | Gate 9 complete | Novelty paragraph auto-generated (FR-METRICS-01) from actual Gate 9 output; paper results draft assembled from real numbers only | FR-METRICS-01, SPEC-01 success criteria |
