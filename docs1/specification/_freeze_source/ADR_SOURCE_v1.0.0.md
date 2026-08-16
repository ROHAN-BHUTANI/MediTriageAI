# MediTriageAI v1.0.0 — Architectural Decision Records
### ADR-001 through ADR-012. No new decisions added — this file only structures the decisions already locked in the FINAL CANDIDATE v1.0.0.

---

### ADR-001 — `meditriage/`'s dependencies on `src/` migrated in audited-risk order, not a blanket cutover
- **Context:** Direct repository audit found `meditriage/` is not actually independent of `src/` today — `models/base_model.py` subclasses `src.model.MediTriageTransformer`; `meditriage/builder/adapters/mtsamples.py` imports `src.specialty_mapping`; `scripts/train_ddp.py` imports from `src.trainer`/`src.config_manager`.
- **Decision:** Migrate `meditriage/`'s inbound dependencies on `src/` module-by-module, in CRITICAL→HIGH→MEDIUM→LOW risk order (per the audited classification), each with a verified parity test before any caller switches over. No blanket "deprecate `src/`" cutover.
- **Consequences:** Slower migration than a rewrite, but no silent change to research results. Requires maintaining both codepaths during transition, with `src/`-sourced numbers visibly footnoted until parity is verified.
- **Alternatives considered:** (a) Full rewrite of `meditriage/` from scratch — rejected, highest risk of silently changing results. (b) Immediate deprecation of `src/` — rejected, `meditriage/` currently depends on it and would break.

---

### ADR-002 — Backbone naming/checkpoint policy: Option B
- **Context:** `XLMRobertaLargeModel` loads `xlm-roberta-base`; `IndicBertModel` loads `google/muril-base-cased`. Class/display names imply different, larger/different models than what's actually used.
- **Decision:** Preserve existing Python class names (avoids breaking checkpoint-loading compatibility). Require exact pretrained checkpoint identifiers everywhere else: documentation, model reports, UI/model registry, experiment artifacts. Prohibit misleading substitutions ("XLM-RoBERTa-base" described as "XLM-RoBERTa-large"; "MuRIL" described as "IndicBERT").
- **Consequences:** No immediate code-breaking rename; requires a new lint (FR-MODEL-02) to enforce checkpoint-identifier accuracy everywhere else. A controlled class-rename may happen later, via normal Change Request, likely bundled with the `src/`→`meditriage/` migration.
- **Alternatives considered:** (Option A) Rename classes immediately to match checkpoint reality — rejected for now, real migration cost against checkpoint compatibility, deferred to later CR. (Option C) Upgrade actual checkpoints to match the names (true XLM-R-large, true IndicBERT) — rejected given stated submission timeline; would require full re-training.

---

### ADR-003 — Embedding initialization: single-anchor clone (current, frozen); multi-anchor mean (future ablation)
- **Context:** `src/vocab_injection.py:56-68` performs 1-to-1 embedding vector cloning from a single canonical anchor. `docs/PAPER_RESULTS_DRAFT.md:178-180` claims mean-of-anchors initialization — contradicted by code.
- **Decision:** Freeze the single-anchor clone as the documented current behavior in SPEC-05. Record multi-anchor mean initialization as a FUTURE-OPTIONAL ablation, not a retrofit into current results. Require paper draft correction before submission.
- **Consequences:** The paper's methodology section must be corrected, which is a real but small editorial task; failing to do so risks a reviewer catching the discrepancy directly. No code change required for this decision itself.
- **Alternatives considered:** Retroactively implementing multi-anchor mean and re-running to match the paper's existing (incorrect) claim — rejected, this would mean the paper drove an implementation change rather than the reverse, and burns time against the submission deadline for a non-essential change.

---

### ADR-004 — Red-flag evaluation methodology
- **Context:** The red-flag mechanism is 8 hardcoded keywords with no ground-truth evaluation dataset anywhere in the repository (confirmed by direct audit).
- **Decision:** Permanently describe the mechanism as "unaudited deterministic heuristic fallback" everywhere it's mentioned. Approve construction of a red-flag evaluation dataset with a defined strata set (genuine red flags, hard negatives, spelling/Hinglish variants, paraphrases, synonyms, indirect descriptions, negation, temporal variation, long/late-occurring inputs, symptom combinations). Construction method, adjudication process, and size remain TBD pending a statistically-justified rationale.
- **Consequences:** No red-flag recall/precision number can be reported until the dataset exists — this is a real gap that must appear in the paper's limitations section as-is, not glossed over.
- **Alternatives considered:** Estimating recall informally from ad hoc testing — rejected, produces an unreliable number that could be mistaken for a validated claim. Skipping the dataset entirely — rejected, leaves a safety-critical mechanism permanently unevaluated with no path to improvement.

---

### ADR-005 — Severity synthetic-data policy
- **Context:** Only ~1.3% of dataset rows have severity labels; 63% of those come from `fedmml_ed_triage`, an LLM-generated (Category C) dataset.
- **Decision:** Include `fedmml_ed_triage` in training, but require every severity result to be reported alongside its synthetic-data share, plus a sensitivity analysis excluding synthetic rows.
- **Consequences:** Severity-task headline numbers must always carry this caveat; a reviewer cannot be shown a clean severity Macro-F1 without also seeing how much of the label signal is synthetic.
- **Alternatives considered:** Excluding `fedmml_ed_triage` entirely — rejected, would leave severity training data even sparser than it already is, likely making the task infeasible. Treating it as equivalent to real ESI labels without caveat — rejected, would misrepresent evidentiary strength to reviewers.

---

### ADR-006 — Evaluation and model-selection policy
- **Context:** McNemar's test is not statistically appropriate for Macro-F1 comparisons (it's designed for paired accuracy/error disagreement); "best," "statistically supported," "most novel," and "most practically useful" had been at risk of being conflated into a single retrospectively-chosen "winner."
- **Decision:** Macro-F1/Weighted-F1 comparisons use paired bootstrap resampling of the metric difference; McNemar's test is restricted to paired accuracy/error-rate comparisons only. Four labels are kept permanently distinct: best performing (primary metric), statistically supported improvement (survives the appropriate test with multiple-comparison correction), most novel architecture (qualitative, human judgment), most practically useful (multidimensional report, not a composite score). Primary/secondary metrics are pre-registered before the DGX benchmark campaign; no retrospective selection.
- **Consequences:** More rigorous, defensible statistics; requires discipline to actually pre-register the metric set in writing before running the benchmark (FR-EVAL-02), rather than choosing after seeing results.
- **Alternatives considered:** Using McNemar's test for all comparisons for simplicity — rejected as statistically incorrect. Defining a single composite "winner" score now — rejected per your explicit Section 8 decision; deferred as a separately-approvable future proposal if ever needed.

---

### ADR-007 — Track C scope boundary
- **Context:** Original scope discussion (v0.1) defaulted to a paper-only deliverable; you subsequently chose full product-grade scope.
- **Decision:** Adopt Track C precisely as "full product-grade research/demo system" — polished dashboard, API, documentation, observability — explicitly not a clinically deployed product, not seeking regulatory clearance, not claiming autonomous-triage capability.
- **Consequences:** Activates SPEC-08/09/13/14/15 as standalone documents rather than deferring them; increases total scope and implementation time relative to a paper-only track, but was your explicit, informed choice given stated goals for the project beyond the paper.
- **Alternatives considered:** Track A (paper + reference implementation only) — the original default, would have kept SPEC-08/09 deferred. Track B (paper + minimal demo) — an intermediate option not chosen.

---

### ADR-008 — DATASET-GATE-01 as a binding pre-training requirement
- **Context:** Per your project history, a prior costly failure mode was training on the wrong/imbalanced dataset without it being caught before compute was spent.
- **Decision:** No official training campaign (LOCAL or DGX) may begin until an 18-item dataset-gate report passes, including language-distribution and class-distribution reports generated and reviewed *before* authorization, not after. Training entrypoints fail loudly, not silently, on a dataset-checksum mismatch.
- **Consequences:** Adds process overhead before every training run, but this overhead is specifically sized to prevent a known, previously-costly failure mode from recurring. Requires new tooling (gate-report generator) to be built (FR-DATA-03, Gate 2).
- **Alternatives considered:** Manual pre-training checklist without automated enforcement — rejected, manual checklists are exactly what failed to catch the problem previously (if the reported historical incident is accurate). Post-hoc auditing only — rejected, catches the problem after compute is already spent.

---

### ADR-009 — Text-provenance taxonomy (SOURCE/A/B/C)
- **Context:** Prior spec versions (including my own earlier drafts) used "synthetic" as an undifferentiated bucket, obscuring that deterministic linguistic augmentation (real text, controlled transform), rule-based templated construction, and LLM generation carry meaningfully different evidentiary strength.
- **Decision:** Add a required, orthogonal `text_provenance` field (`SOURCE`/`A`/`B`/`C`) to the canonical schema, distinct from the existing label-provenance field (`DIRECT`/`MAPPED`/`INFERRED`).
- **Consequences:** Requires a retroactive audit of every AUXILIARY dataset's augmentation to assign it a category — real, non-trivial work, but resolves a genuine ambiguity that would otherwise undermine the paper's evidentiary claims.
- **Alternatives considered:** Leaving "synthetic" as a single flag — rejected, insufficiently precise for a paper that needs to defend its evidentiary basis to reviewers.

---

### ADR-010 — 20-dimension Multilingual Robustness Framework
- **Context:** "MediTriageAI handles Hinglish" risked being an overclaim relative to what's actually demonstrated — a controlled phonetic/orthographic layer, not general compositional/code-switched semantic understanding.
- **Decision:** Adopt the 20-dimension framework as the required structure for describing multilingual capability, with each dimension requiring a five-link evidentiary chain (generation → training exposure → validation → evaluation metric → acceptance criterion) before it can be claimed as demonstrated. Adopting the framework does not itself claim any dimension is complete.
- **Consequences:** Most dimensions currently sit at UNKNOWN or PARTIAL pending a real audit (Gate 4) — this is accurate, not a failure of the framework; the framework's value is precisely in making that visible rather than letting a vague "Hinglish support" claim stand unexamined.
- **Alternatives considered:** Continuing to describe multilingual capability in general terms without a dimension-by-dimension breakdown — rejected, this is exactly the overclaim risk the framework exists to prevent.

---

### ADR-011 — Practical usefulness reported as a multidimensional report, not a composite score
- **Context:** A single "most practically useful model" composite score would require an undefined, arbitrary utility function (weighting accuracy vs. latency vs. size vs. calibration).
- **Decision:** Report practical usefulness as a multidimensional report (routing quality, severity quality, dangerous error rate, calibration, multilingual robustness, OOD behavior, explainability, latency, resource requirements, failure modes) rather than a single score, at this stage.
- **Consequences:** No single "winner" can be declared on practical grounds without a reader forming their own judgment from the report — arguably more honest, but less immediately actionable as a marketing-style claim.
- **Alternatives considered:** Defining a composite score now with placeholder weights — rejected, an arbitrary weighting would be indistinguishable from post-hoc metric selection dressed up as objectivity.

---

### ADR-012 — Safety-gatekeeper thresholds default to TBD
- **Context:** A proposed `ECE ≤ 0.08` threshold had no cited scientific basis — a plausible-sounding round number with no justification.
- **Decision:** No numerical safety threshold (calibration, dangerous-confusion-rate cutoff, etc.) is frozen as normative unless supported by (A) a cited authoritative source or (B) an explicitly documented empirical methodology. Until then, all such values are reported, never used as pass/fail gates, and marked TBD / NOT CLINICALLY VALIDATED.
- **Consequences:** No safety-related pass/fail automation can be built until a threshold is properly justified — this is a deliberate constraint, not an oversight, and prevents a fabricated-looking threshold from ever appearing in the frozen spec or the paper.
- **Alternatives considered:** Adopting a plausible literature-adjacent number without direct citation — rejected as functionally indistinguishable from inventing evidence.