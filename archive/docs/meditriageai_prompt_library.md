# MediTriageAI — Execution Prompt Library (Antigravity / Gemini Pro)

**How to use this:** Run these one at a time, in order. Don't fire the next prompt until you've personally verified the exit criteria of the current one — reading the actual output, not just skimming that it "ran." Each phase assumes the previous phase's artifacts exist on disk. Paste the prompt as-is; adjust bracketed placeholders.

Priority logic: **Paper-critical work (dataset → baselines → transformers → stats → writing) comes before product polish (backend/frontend/mobile).** A reviewer rejects on weak methodology, not on missing a Flutter app.

---

## PHASE 0 — Freeze the Architecture (no code)

```
Act as the Lead Research Engineer auditing the MediTriageAI repository.

Do not write or generate any code in this response.

1. List every component that currently exists (dataset scripts, model code,
   dashboard, docs) based on the repo structure.
2. For each component, state: DONE / PARTIAL / MISSING / UNVERIFIED.
3. Identify every open architectural decision that is still ambiguous
   (e.g., label taxonomy version, train/val/test split ratios, which
   transformer variants, loss weighting scheme).
4. Propose a single frozen decision for each ambiguity, with one sentence
   of justification.
5. Output a FINAL_ARCHITECTURE.md file capturing all frozen decisions.

Exit criteria: I can read FINAL_ARCHITECTURE.md and there is nothing left
that will change once training starts.
```

**Exit check:** You can name every hyperparameter, split ratio, and label schema without saying "we'll decide later."

---

## PHASE 1 — Dataset Reality Check (verify, don't rebuild)

```
Act as the Data Engineering Lead.

Using the existing dataset pipeline (MTSamples ingestion + Hinglish
perturbation + leakage-safe split):

1. Run the full pipeline end-to-end and report actual row counts at every
   stage: raw → sanitized → perturbed → train/val/test.
2. Verify no tracking ID appears in more than one split (leakage check).
   Print the count of duplicates found (should be zero).
3. Print the class distribution (severity labels, specialist labels) for
   each split as a table.
4. Flag any split with fewer than 30 examples per class — this is a
   statistical validity risk, not just a formatting issue.
5. Output DATASET_REPORT.md with these numbers and a plain-language
   verdict: "ready for training" or "needs rebalancing, here's why."

Do not modify the pipeline yet. This is a verification pass only.
```

**Exit check:** You have real numbers, not "should work." If class counts are too small, fix that *before* Phase 2 — training on a broken split wastes every downstream hour.

---

## PHASE 2 — Baseline Models (cheap, fast, and necessary)

```
Act as the ML Engineer.

Train these baselines on the frozen dataset split from DATASET_REPORT.md:
- TF-IDF + Logistic Regression
- TF-IDF + Linear SVM
- TF-IDF + Random Forest

For each: report accuracy, macro-F1, per-class precision/recall, and a
confusion matrix (as an image and as raw numbers).

Output BASELINE_RESULTS.md with a comparison table across all three,
and one paragraph: "here is why a transformer should beat this baseline,
specifically on which failure modes."

Do not train any transformer in this step.
```

**Exit check:** You have a real number to beat. This is what makes "our transformer improves macro-F1 by X points" a defensible sentence in the paper instead of a vibe.

---

## PHASE 3 — Transformer Training (one model at a time, verified)

Run this prompt **separately for each model** — don't batch all four in one shot, or you won't be able to debug which one failed or catch a silent numerical issue.

```
Act as the ML Engineer.

Train [MODEL_NAME: e.g. XLM-RoBERTa-large] on the frozen dataset split,
using the dual-head architecture (severity head + specialist head) and
weighted joint loss already implemented in the repo.

Fixed seed: 42. Do not change hyperparameters from FINAL_ARCHITECTURE.md
without flagging it first.

Report after training:
- Training loss curve and validation loss curve (as plot + raw values)
- Macro-F1, accuracy, per-class precision/recall for BOTH heads
- Confusion matrix for severity head
- Training time and hardware used
- Whether validation loss diverged from training loss (overfitting check)

Save the checkpoint. Output [MODEL_NAME]_RESULTS.md.

If any metric looks implausibly high (e.g. >98% F1 on a hard multilingual
task), flag it explicitly as a possible leakage or evaluation bug before
reporting it as a result.
```

Repeat for mBERT, DistilBERT (or IndicBERT), etc. — however many you can actually afford in compute/time. **Two well-validated transformers beat four rushed ones.**

**Exit check:** Every number in every _RESULTS.md file is something you could explain and defend out loud to a professor, with no "the model just said this."

---

## PHASE 4 — Model Comparison + Statistical Validation

```
Act as the Research Scientist.

Using BASELINE_RESULTS.md and all [MODEL]_RESULTS.md files:

1. Build one master comparison table: all baselines + all transformers,
   same metrics, same formatting.
2. Run McNemar's test comparing the best baseline vs. the best transformer.
   Report the test statistic and p-value.
3. Compute a bootstrap 95% confidence interval on macro-F1 for the best
   model (1000 resamples).
4. Break down accuracy by language (English / Hindi / Hinglish /
   code-mixed) for the best model — this is likely your strongest
   novelty angle.
5. Output RESULTS_MASTER.md with all tables, the statistical tests, and
   one paragraph stating what is and isn't statistically significant.

Do not claim significance without the test backing it.
```

**Exit check:** You can say "the improvement is statistically significant (p < 0.05)" or "the improvement is not statistically significant, here's what that means for our claims" — either way, honestly.

---

## PHASE 5 — Novelty Check (reviewer mindset, before you write)

```
Act as a skeptical IEEE/Springer reviewer who has read 200 similar
clinical NLP triage papers.

Given RESULTS_MASTER.md and FINAL_ARCHITECTURE.md:

1. What does this work contribute that isn't already covered by existing
   multilingual clinical NLP or Hinglish NLP papers? Be specific — name
   the closest prior work you'd expect a reviewer to cite against us.
2. Is the contribution the dataset, the architecture, the language
   analysis, or the calibration/robustness angle? Pick the ONE strongest
   angle — don't hedge across four weak ones.
3. What's the single biggest weakness a reviewer would flag first?
4. What is the minimum additional experiment needed to address that
   weakness, given I have limited compute and time?

Be harsh. I'd rather hear this now than at peer review.
```

**Exit check:** You can state your paper's contribution in one sentence, and you know your weakest point before a reviewer does.

---

## PHASE 6 — Paper Draft (incremental, section by section)

```
Act as a co-author writing an IEEE-format paper on MediTriageAI.

Write ONLY the [SECTION: e.g. Methodology] section, based strictly on
FINAL_ARCHITECTURE.md, DATASET_REPORT.md, and RESULTS_MASTER.md.

Rules:
- Every claim must map to a number that actually exists in those files.
- No invented citations. If a citation is needed, mark it
  [CITATION NEEDED: topic] rather than fabricating one — I will find
  real papers via search separately.
- No inflated language ("groundbreaking," "revolutionary"). IEEE tone:
  precise, hedged, evidence-first.
- Include limitations honestly — reviewers trust papers that admit them.

Output as LaTeX using the standard IEEE conference template structure.
```

Run once per section: Abstract, Introduction, Related Work, Methodology, Experiments, Results, Discussion/Limitations, Conclusion. For Related Work specifically, use live web search for real papers — don't let it invent citations from memory.

---

## PHASE 7 — Reviewer Attack (do this before submission, not after)

```
Act as three independent IEEE reviewers with different specialties:
(1) an NLP methodologist, (2) a clinical informatics expert,
(3) a statistics/evaluation expert.

Read the full paper draft. Each reviewer writes:
- 3 strengths
- 3 concrete weaknesses (not vague — point to specific sentences/tables)
- A recommendation: Accept / Minor Revision / Major Revision / Reject

Then, as the response, propose a concrete fix for every weakness raised,
ranked by effort required (low/medium/high).
```

---

## STRETCH PHASES (only after the above is genuinely solid)

These are worth doing **if time allows** — but don't let them eat time that should go to Phases 0–7. A working inference API + one clean demo dashboard is enough for a hackathon/portfolio; a full Flutter app + iOS build is a multi-week project on its own.

```
[Backend MVP]
Act as a backend engineer. Build a FastAPI service exposing a single
/predict endpoint that loads the best-performing checkpoint from Phase 3
and returns severity + specialist + confidence. Include basic auth,
input validation, and OpenAPI docs. Nothing more yet.

[Dashboard]
Act as a frontend engineer. Build a single-page research dashboard
(model comparison table, confusion matrix heatmap, language-wise
accuracy chart) wired to RESULTS_MASTER.md as static JSON. No backend
dependency needed for this — it's for demos and the paper, not production.

[Mobile — only if genuinely needed for a hackathon deliverable]
Scope down to a single-screen React Native or Flutter demo: symptom
input → prediction → severity banner. Skip offline queue, OCR, and
multi-language voice input unless a specific deadline requires them.
```

---

## Running this across sessions
Since this spans many phases, at the end of each phase ask Gemini to output a short **STATE_SUMMARY.md** (what's frozen, what's verified, what's next) and paste that back in as context when you start the next session — don't rely on one infinite chat thread to hold all of this reliably.
