# MediTriageAI v1.0.0 — PRE-FREEZE AMENDMENTS
### Status: amendments to the FINAL CANDIDATE, still pre-freeze. Not a redesign — every item below is a targeted correction to the document already approved in substance.

---

## AMENDMENT 1 — Language audit deferred to post-freeze (not a freeze blocker)

Section 5a of the candidate is revised: the historical language-distribution audit remains **UNKNOWN/PENDING** at freeze time. It is **not required before specification freeze.**

It becomes **GATE 1** (see Amendment 5) — the first post-freeze execution task — and **must run before any new official training campaign or before any reliance on previous multilingual benchmark claims.** If the historical artifact fails DATASET-GATE-01, it is classified **NOT ELIGIBLE** for final benchmark use, per the binding rule already established. This does not change anything substantive from the candidate — it only removes the ambiguity about whether the audit blocks freeze. It does not.

---

## AMENDMENT 2 — Freeze vs. implementation distinction (explicit, binding statement)

**Freezing v1.0.0 means freezing the requirements, architecture, governance model, acceptance criteria, and change-control contract. It does not mean claiming any requirement is already implemented.**

Every requirement in the frozen specification carries one of the following implementation-status values, and freezing the spec does not change any of these values — it only locks *what the requirement is*, not *whether it's done*:

- **IMPLEMENTED** — exists in the repository today, verified.
- **PARTIAL** — exists in some form but doesn't meet the full acceptance criterion yet.
- **MISSING** — does not exist yet; a target and acceptance criterion are specified regardless.
- **UNKNOWN** — existence/correctness not yet verified; must be checked, not assumed either way.
- **MIGRATION** — mid-transition from a `src/` implementation to a `meditriage/` implementation.
- **RESEARCH-EXPERIMENTAL** — implemented but not validated to production standard (E-PATH is the standing example).
- **TBD** — deliberately unset pending a documented justification (safety thresholds, red-flag dataset size).

This statement itself is now part of the frozen document's governing text — it is the rule that prevents "we froze the spec" from ever being misread as "the system is done."

---

## AMENDMENT 3 — Expanded traceability matrix (every normative FR-ID)

Replaces the representative matrix in the candidate's Section 14. Provided as a standalone file — see `MediTriageAI_v1.0.0_TRACEABILITY.md`, presented alongside this document. It maps every FR-ID to: SPEC section → implementation location/target → verification method → acceptance criterion → current status, with no FR-ID left unmapped and no status left implicit.

---

## AMENDMENT 4 — ADRs with Context/Decision/Consequence/Alternatives

No new ADRs added — the existing ADR-001 through ADR-012 list from the candidate is confirmed sufficient for freeze, per your instruction. Each now has the compact structure GSD/Ralph need to act without re-deriving rationale. Provided as a standalone file — see `MediTriageAI_v1.0.0_ADRs.md`.

---

## AMENDMENT 5 — Explicit post-freeze implementation gates

These gates are now part of the frozen governance contract. **GSD/Ralph must not skip a gate or begin a later gate before an earlier one's exit criterion is met**, except where explicitly marked parallelizable.

| Gate | Name | Entry criterion | Exit criterion | Primary FR-IDs / SPEC sections |
|---|---|---|---|---|
| **0** | Specification integrity | Freeze authorized | Frozen file checksums (Amendment 9's `MANIFEST.json`) verified against on-disk copy before any other gate proceeds | Freeze package itself |
| **1** | Historical dataset language audit | Gate 0 passed | Language-distribution audit report (Section 5a format) produced and filed; historical artifact classified ELIGIBLE or NOT ELIGIBLE against DATASET-GATE-01 | FR-DATA-03 (partial), Section 5a |
| **2** | Dataset Gate implementation | Gate 0 passed (parallelizable with Gate 1) | DATASET-GATE-01 mechanism itself exists in code: gate-report generator, checksum verification, fail-loudly training entrypoint, text-provenance field wired through | FR-DATA-03, FR-TEXT-01 |
| **3** | Priority-1 `src→meditriage` parity | Gate 0 passed (parallelizable with Gates 1–2) | `src/model.py`, `src/trainer.py`, `src/dataset.py`, `src/schema.py` ported with verified numerical/behavioral parity; `config.get_hash()` existence resolved (implemented if missing) | FR-EVAL-01, FR-EVAL-03, SPEC-03 Priority-1 table |
| **4** | Multilingual robustness audit | Gate 0 passed (parallelizable) | All 20 dimensions in the Robustness Matrix (Section 6) have a determined status (not UNKNOWN) with evidence for each of the five required links | FR-UX-03 |
| **5** | Red-flag evaluation dataset | Gate 0 passed (parallelizable) | Construction method, adjudication methodology, and statistically-justified size are all determined and documented; dataset built | SPEC-07 strata table |
| **6** | Production implementation | Gates 1–5 substantially complete for the modules each touches | All remaining MISSING FR-IDs in the traceability matrix implemented (disclaimer enforcement, novelty-paragraph generator, API field separation, naming-policy lint, `/version` endpoint, doc-linter) | Full traceability matrix |
| **7** | Full test suite | Gate 6 complete | 444+ existing tests plus all new safety/adversarial/snapshot/lint tests pass | SPEC-10 |
| **8** | DGX canonical training | Gates 2, 3, 7 passed; Gate 1 result reviewed | Training executed under a DATASET-GATE-01-passed dataset, against pre-registered primary metrics (FR-EVAL-02 artifact filed *before* this gate opens) | FR-EVAL-02, DGX Execution Contract |
| **9** | DGX evaluation / statistical verification | Gate 8 complete | Full evaluation run: paired bootstrap for Macro-F1 comparisons, McNemar for accuracy comparisons, calibration/safety/multilingual/OOD metrics all reported | SPEC-06 methodology |
| **10** | Final research artifact generation | Gate 9 complete | Novelty paragraph auto-generated (FR-METRICS-01) from actual Gate 9 output; paper results draft assembled from real numbers only | FR-METRICS-01, SPEC-01 success criteria |

---

## AMENDMENT 6 — Ralph: single-bounded-objective constraint

Ralph is **not permitted to autonomously execute across multiple gates in one run.** Every Ralph invocation must be scoped with, at minimum:

```
objective: <one bounded, atomic task>
fr_or_spec_reference: <exact FR-ID or SPEC section>
gate: <which gate from Amendment 5 this belongs to — Ralph may not act outside its declared gate>
allowed_files: [<explicit file/directory allowlist>]
verification_command: <exact command to run, e.g. `pytest tests/test_X.py`>
stop_condition: <what makes this run done or blocked>
expected_acceptance_criterion: <copied verbatim from the traceability matrix>
```

A Ralph run that touches files outside its declared allowlist, or whose stated objective spans more than one gate, is out of contract regardless of whether its output looks correct.

---

## AMENDMENT 7 — CodeRabbit: independent review layer, not an acceptance authority

CodeRabbit's "approve" result is **not equivalent to passing the specification's acceptance criteria.** CodeRabbit performs code-quality/style/security-pattern review; the traceability matrix's verification method (test commands, checksum checks, manual reviews as specified) remains the sole authority on whether a requirement is satisfied. A PR may be CodeRabbit-approved and still fail its FR's acceptance criterion, and vice versa in principle (though a failing CodeRabbit review should still block merge per the existing gates in SPEC-10/23).

---

## AMENDMENT 8 — Exact DGX execution sequence

Replaces the general DGX Execution Contract description in the candidate with this explicit ordered sequence, which the DGX runner must follow with no steps skipped or reordered:

```
1. FROZEN SPEC CHECKSUM       — verify local frozen copy against MANIFEST.json (Gate 0)
2. DATASET AUDIT               — language-distribution + full DATASET-GATE-01 report (Gates 1–2)
3. DATASET-GATE-01             — pass/fail determination; halt here if FAIL
4. TRAINING CONFIG VALIDATION  — config hash computed and matched against declared value (Gate 3 dependency)
5. TRAINING                    — executed only if steps 1–4 all passed (Gate 8)
6. EVALUATION                  — full metric suite per SPEC-06 (Gate 9)
7. STATISTICAL VERIFICATION    — paired bootstrap / McNemar per the approved methodology (Gate 9)
8. ARTIFACT MANIFEST           — every output tagged with spec version + dataset checksum (SPEC-13)
9. REPRODUCIBILITY CHECK       — re-run verification confirming the artifact manifest's claims are reproducible from the recorded checksums
```

Any failure at steps 1–4 halts the sequence before compute is spent on steps 5–9 — this is the direct structural fix for the "trained on the wrong dataset" failure mode.

---

## AMENDMENT 9 — Freeze package location and required contents

Canonical repository location: **`docs/specification/frozen/v1.0.0/`**

Required contents:
```
docs/specification/frozen/v1.0.0/
  SPECIFICATION.md      <- the full consolidated v1.0.0 text (candidate document, as amended)
  TRACEABILITY.md        <- Amendment 3's full FR-ID matrix
  ADRs/                  <- Amendment 4's ADR-001 through ADR-012, one file each or a combined index
  RISK_REGISTER.md        <- Section 15's consolidated risk register
  MANIFEST.json           <- SHA-256 of each file above + freeze metadata
  SHA256SUMS              <- flat checksum list, standard format, for quick external verification
  FREEZE_RECORD.md        <- Amendment 10's content
```

Also create: **`docs/specification/CURRENT.md`** — a short pointer file containing only a reference to `frozen/v1.0.0/` as the currently authoritative specification version (this is what future versions update, rather than anyone needing to guess which version is current).

**Binding constraint:** the freeze operation writes only to `docs/specification/`. It does not modify any source implementation file. This is a documentation/governance commit, not a code change.

---

## AMENDMENT 10 — `FREEZE_RECORD.md` required content

```markdown
# MediTriageAI Specification Freeze Record

Specification version: v1.0.0
Git commit SHA at freeze time: <to be filled by Antigravity at freeze execution>
Freeze timestamp (UTC): <to be filled at freeze execution>
Manifest checksum (MANIFEST.json SHA-256): <to be filled at freeze execution>

## Authoritative Status
This document certifies that MediTriageAI Specification Baseline v1.0.0, located at
docs/specification/frozen/v1.0.0/, is the authoritative specification baseline for
requirements, architecture, governance, acceptance criteria, and change control.

## Implementation Status Disclaimer
Freezing this specification does NOT imply that any requirement described within it
is already implemented. Implementation status for every requirement is tracked
independently in TRACEABILITY.md using the status values: IMPLEMENTED, PARTIAL,
MISSING, UNKNOWN, MIGRATION, RESEARCH-EXPERIMENTAL, TBD. Freeze fixes *what* each
requirement means, not whether it is currently satisfied.

## Change Control
Any modification to a requirement, interface, architectural decision, naming policy,
or acceptance criterion described in this frozen baseline requires an explicit Change
Request per SPEC-15, human approval, and a new specification version. No autonomous
agent (GSD, Ralph, CodeRabbit, or any DGX execution) may alter this baseline directly.
```

---

## AMENDMENT 11 — Terminology constraint on the word "frozen"

**"Frozen" describes only the approved specification baseline** (`docs/specification/frozen/v1.0.0/`) unless explicitly qualified otherwise. It must never be used, anywhere in documentation, code comments, dashboard copy, or the paper, to describe:
- currently implemented code ("the frozen implementation" — wrong; use "current implementation, targeting frozen SPEC-0X")
- the current dataset ("frozen dataset" — only correct if explicitly qualified as a specific checksummed artifact, e.g. "frozen dataset (checksum `<hash>`)," not the general repository dataset state)
- current model weights ("frozen checkpoint" — only correct with an explicit checkpoint identifier/hash attached)
- current benchmark results ("frozen benchmark configuration" — only correct with the specific config's hash attached)

Unqualified "frozen" always means the specification document, nothing else.

---

## PRE-FREEZE STATUS

| # | Item | Status |
|---|---|---|
| 1 | Language audit deferred to Gate 1, non-blocking for freeze | **APPROVED** |
| 2 | Freeze ≠ implementation, status taxonomy binding | **APPROVED** |
| 3 | Full traceability matrix | **APPROVED** — see `MediTriageAI_v1.0.0_TRACEABILITY.md` |
| 4 | ADRs with Context/Decision/Consequence/Alternatives | **APPROVED** — see `MediTriageAI_v1.0.0_ADRs.md` |
| 5 | Gates 0–10 defined as binding post-freeze sequence | **APPROVED** |
| 6 | Ralph single-bounded-objective contract | **APPROVED** |
| 7 | CodeRabbit independent-review clarification | **APPROVED** |
| 8 | Exact DGX 9-step sequence | **APPROVED** |
| 9 | Freeze package location + contents (`docs/specification/frozen/v1.0.0/`, `CURRENT.md`) | **APPROVED** |
| 10 | `FREEZE_RECORD.md` template | **APPROVED** |
| 11 | "Frozen" terminology constraint | **APPROVED** |
| — | Historical language-distribution audit result | **UNKNOWN — PENDING POST-FREEZE (Gate 1)** |
| — | All Gate 1–10 implementation work | **PENDING POST-FREEZE EXECUTION** |
| — | Safety-gatekeeper threshold values | **TBD** (unchanged from candidate) |
| — | Red-flag dataset size | **TBD** (unchanged from candidate) |

No further redesign performed. This amendment set, combined with the FINAL CANDIDATE v1.0.0 document, the traceability matrix, and the ADR file, is what's ready for your authorization to execute the actual Antigravity freeze into `docs/specification/frozen/v1.0.0/`.