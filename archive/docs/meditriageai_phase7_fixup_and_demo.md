# MediTriageAI — Phase 7 Fix-Up + Product Demo Packaging

Run these in order in Antigravity, in the same session that has your state summary as context. Don't skip to the demo prompts until the fix-up prompts are done — an honest, smaller result set beats an impressive-looking but indefensible one.

---

## STEP 1 — Fix the severity baseline leakage

```
Act as the ML Engineer auditing a possible label leakage issue.

Our severity labels were generated using a regex/keyword heuristic scanner
(documented as low-confidence/unvalidated). Our TF-IDF + Random Forest
baseline scores 93.70% macro-F1 on severity, while XLM-RoBERTa-large scores
32.48% macro-F1 on the same task and same-family test data.

1. Extract the top TF-IDF features the Random Forest baseline relies on
   most heavily for its severity predictions (feature importances).
2. Compare these features against the actual keyword list used in the
   regex severity heuristic that generated the labels.
3. Report the overlap percentage and give a verdict: is the baseline's
   93.70% score primarily explained by it re-detecting the same keywords
   used to generate the label (circularity), or is it learning something
   genuinely predictive beyond the heuristic?
4. Output LEAKAGE_AUDIT.md with the finding stated plainly, and a
   recommendation: either (a) exclude this baseline comparison from the
   paper's headline results and document it as a known limitation, or
   (b) get real clinician-annotated severity labels for at least a
   held-out validation subset so the transformer can be judged against
   ground truth instead of against the heuristic's own logic.

Do not soften this finding — if it's circular, say so clearly.
```

**Why this matters:** if you write "our baseline achieves 93.7% macro-F1" in the paper without this audit, and a reviewer traces it back to the label heuristic, it reads as either sloppy or dishonest. Naming it yourself, with a documented audit, reads as rigorous.

---

## STEP 2 — Full-test-set transformer evaluation (fix the sample-size mismatch)

```
Act as the ML Engineer.

Re-evaluate XLM-RoBERTa-large and mBERT on the FULL test split
(2,000 rows), not the 160-sample subset used previously. Use the saved
checkpoints — no retraining needed, this is an evaluation-only pass.

Report for both models, on both heads (severity, specialist):
- Macro-F1, accuracy, per-class precision/recall
- Confusion matrix
- McNemar's test comparing best transformer vs. best classical baseline,
  now on the SAME 2,000-row test set for both sides (eliminates the
  sample-size mismatch flagged in the previous session)
- Bootstrap 95% CI on macro-F1, recomputed on the full test set
- Language-wise breakdown (English vs. Hinglish vs. code-mixed) on the
  full test set, with per-language sample counts shown explicitly next
  to each accuracy number — small-n subsets must be labeled as such

If any full-test-set number changes meaningfully from the 160-sample
result, flag that explicitly rather than quietly replacing it.

Output RESULTS_MASTER_FULL.md.
```

**Why this matters:** your current McNemar significance test compared models evaluated on different sample sizes — that invalidates the comparison. This step is the one you already correctly identified was needed; this just runs it.

---

## STEP 3 — Rotate the exposed credential

```
Act as a backend engineer doing a security pass before any demo or repo push.

In scripts/serve_api.py:
1. Remove the hardcoded username/password from source code.
2. Load credentials from environment variables instead
   (e.g. os.environ["MEDITRIAGE_API_USER"], os.environ["MEDITRIAGE_API_PASS"]).
3. Add a .env.example file showing the expected variable names without
   real values, and confirm .env is in .gitignore.
4. Confirm no other file in the repo contains a plaintext credential —
   grep for common patterns (password, secret, api_key, token) and report
   any other hits.

Output a one-paragraph confirmation that the repo is safe to share or screen-record.
```

**Why this matters:** you're about to package this for demos, screen recordings, and possibly a public repo — do this before that, not after.

---

## STEP 4 — Package the honest demo

Once Steps 1–3 are done, run this to build what you'll actually show:

```
Act as a product engineer preparing MediTriageAI for a live demo
(hackathon judges / academic panel).

Using RESULTS_MASTER_FULL.md and LEAKAGE_AUDIT.md as the source of truth:

1. Update the dashboard (dashboard_web/) to show:
   - Model comparison table using ONLY full-test-set numbers
   - A clearly labeled "known limitation" panel showing the severity
     baseline circularity finding from the audit, framed as "identified
     and addressed" rather than hidden
   - The McNemar + bootstrap CI results with sample sizes shown
   - Language-wise breakdown WITH sample counts visible per language
2. Verify the inference API's /predict endpoint works end-to-end with
   3-5 example inputs (mix of English, Hindi, Hinglish), and print the
   actual request/response pairs so I can screenshot them for the demo.
3. Generate a DEMO_SCRIPT.md: a 5-minute walkthrough script covering
   (a) the problem and why it's real (cite eSanjeevani/ASHABot scale),
   (b) the dataset + dual-head architecture, (c) one live prediction call,
   (d) the honest results — including the specialist-routing challenge
   as an open problem, not a hidden weakness, (e) what's next.

Do not inflate any number. Where results are early-stage (e.g. specialist
routing near-chance), frame it as "here's the harder problem we're solving
next" rather than omitting it.
```

---

## What "showing everything" should actually look like

For a hackathon or academic panel, lead with:
1. **The working pipeline, live** — dataset → training → inference API → dashboard, actually running, not slides.
2. **The statistical rigor** — McNemar's test and bootstrap CI are things almost no undergrad project has; say so explicitly, it's a genuine differentiator.
3. **The honesty** — naming the severity-label circularity as something you caught and are fixing is more impressive to any technical judge than hiding it would be. It signals research maturity.
4. **The open problem** — specialist routing near-chance performance, framed as "this is the hard part, here's our plan" (more data, better label quality, or a retrieval-augmented specialist-matching layer) is a legitimate, fundable-sounding research direction — not a failure to apologize for.
5. **The real-world grounding** — eSanjeevani and ASHABot numbers, to show this isn't a toy problem.

Don't lead with the 93.7% severity baseline number — after the audit, it's not a number you want on your first slide.
