# MediTriageAI — CRITICAL: Fix Training Data Starvation Before Continuing

Run this before writing any more of the paper or finalizing the demo. The
full-test-set bootstrap CI collapse ([0.50%, 0.75%] macro-F1, down from
[11.54%, 15.96%] on the 160-sample subset) indicates the transformers were
trained on far too little data to generalize. Everything downstream of
training is currently invalid until this is fixed.

---

## STEP 0 — Confirm the root cause before retraining anything

```
Act as the ML Engineer diagnosing a performance collapse.

Our transformers (XLM-RoBERTa-large, mBERT) were trained on a 160-sample
split, but the full training set contains 15,996 rows. On the full
1,999-row test set, mBERT's macro-F1 bootstrap CI collapsed to
[0.50%, 0.75%], versus [11.54%, 15.96%] on a 160-sample test subset.

1. Confirm exactly how many training rows were actually used in the
   Phase 3 training run — check train.py's data loader / config, not
   just the log message.
2. Print the predicted label distribution on the full test set. If the
   model is predicting one or two classes almost exclusively, confirm
   that explicitly — this would indicate collapse to a degenerate
   baseline (majority-class prediction) rather than genuine learning.
3. Reconcile the test set row count discrepancy (2,000 in one document,
   1,999 in another) — identify where the off-by-one or filtering step
   is happening.
4. Output DIAGNOSIS.md stating plainly: was this a resource-constrained
   under-training issue, a data loader bug, or something else?

Do not retrain yet. This step is diagnosis only.
```

---

## STEP 1 — Retrain on a resource-feasible but adequate data size

```
Act as the ML Engineer, working within CPU-only local resource
constraints (no GPU available).

Retrain mBERT and XLM-RoBERTa-large (or, if XLM-R-large is too slow on
CPU, substitute distilbert-base-multilingual-cased as a lighter
alternative for this pass) using:
- A training subset of at least 3,000–5,000 rows (not 160) — pick the
  largest size that completes in a reasonable time budget on this
  machine. If even that's infeasible on CPU, say so explicitly with an
  estimated time cost, rather than silently defaulting back to a tiny subset.
- Gradient accumulation to simulate a larger effective batch size if
  memory-constrained.
- Mixed precision (fp16) if supported on this CPU/setup to speed training.
- The same fixed seed (42) and same frozen hyperparameters from
  FINAL_ARCHITECTURE.md — do not silently change loss weights or
  learning rates without flagging it.

Report training time, final training/validation loss curves, and confirm
convergence (loss decreasing and stabilizing, not still dropping sharply
at the last epoch — if it's still dropping, more epochs are needed).

Output RETRAIN_LOG.md with these details before re-evaluating.
```

---

## STEP 2 — Re-run full-test-set evaluation and statistics (mandatory redo)

```
Act as the ML Engineer.

Using the newly retrained checkpoints from Step 1, redo the full
evaluation pipeline exactly as before:
- Full 1,999(or corrected)-row test set macro-F1, accuracy, per-class
  precision/recall, confusion matrix, for both heads
- McNemar's test vs. the best classical baseline, same test set for both
- Bootstrap 95% CI on macro-F1, full test set
- Language-wise breakdown (English/Hindi/Hinglish) WITH sample counts
  shown per language subgroup

Compare these new numbers against RESULTS_MASTER_FULL.md's numbers
directly, and report the delta. If the bootstrap CI no longer collapses
to near-zero, that confirms the original run was a data-starvation issue.
If it still collapses, escalate — that would mean the problem is
architectural or in the data pipeline, not just training set size.

Output RESULTS_MASTER_FULL_V2.md — this replaces the previous one as the
paper's source of truth. Do not blend numbers from both versions in the
same table.
```

---

## Only after Step 2 confirms real, defensible numbers:

- Resume the **Matched Medium-Scale Evaluation** (Linear SVM vs. DistilBERT-multilingual on the same subset) — this was correctly identified as a good next step, but it should use the same properly-sized training data as Step 1, not repeat the 160-sample mistake.
- Resume the **clinician-annotated subset** work — this is the right long-term fix for the severity-label circularity and remains valuable regardless of the retraining outcome.
- Only then continue **Phase 6** (Abstract, Introduction, Related Work, Experiments, Results, Discussion, Conclusion) — every number in the Results section should trace to RESULTS_MASTER_FULL_V2.md, not the earlier invalidated file.

**Do not update the live dashboard or DEMO_SCRIPT.md with the old numbers in the meantime** — if you have a demo scheduled before Step 2 finishes, present the API/pipeline as working (it is) but hold off on quoting the specialist-routing macro-F1 number until it's re-verified.
