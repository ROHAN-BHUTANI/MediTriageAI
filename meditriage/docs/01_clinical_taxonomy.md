# MediTriageAI — Clinical Severity Taxonomy & Specialist Routing Schema

**Status:** Draft v0.1 — heuristic labels, NOT clinically validated. Unvalidated/low-confidence by design (see §5).
**Scope:** Defines the two label spaces used by the dual-head classifier: (1) clinical severity, (2) specialist department routing.

---

## 1. Why ESI as the severity backbone

We anchor the severity scale on the **Emergency Severity Index (ESI)**, a five-level ED triage algorithm
originally developed in 1998 by emergency physicians Wuerz and Eitel together with nurses Gilboy, Tanabe,
and Travers, and currently maintained by the Emergency Nurses Association. ESI has been validated across
multiple versions: ESI v3 was shown in a retrospective cohort of 403 ED patients to predict resource
consumption, with mean resource use decreasing monotonically from Level 1 (5.0 resources) to Level 5
(0.2 resources). ESI v5 (2025) added explicit vital-sign-based uptriage checks at levels 3–5 to reduce
undertriage risk.

We use ESI as a **conceptual anchor, not a clinical instrument**. ESI is designed for prospective in-person
triage by a trained nurse who can observe vital signs and ask follow-up questions. Our task — inferring
severity from retrospective, free-text clinical narrative (MTSamples transcriptions) and synthetic Hinglish
patient self-reports — is a different and strictly harder problem. We therefore borrow ESI's 5-tier
*structure* and resource-based intuition, but our labels are a text-derived proxy, not an ESI score.

## 2. The 5-level severity taxonomy

| Level | Name | ESI analogue | Definition for this project | Example signal |
|---|---|---|---|---|
| **S1** | Resuscitation / Immediate | ESI-1 | Life-threatening; would require immediate intervention if presenting now. Cardiac/respiratory arrest, unresponsive, severe hemorrhage, anaphylaxis. | "in cardiac arrest", "not breathing", "massive hemorrhage", "unresponsive" |
| **S2** | Emergent | ESI-2 | High-risk situation; should not wait. Severe pain, altered mental status, stroke-like symptoms, suspected MI, severe dyspnea. | "severe chest pain", "sudden weakness one side", "worst headache of my life" |
| **S3** | Urgent | ESI-3 | Needs prompt evaluation and multiple resources, but not immediately life-threatening. Moderate pain, persistent vomiting, high fever with comorbidity. | "persistent high fever", "moderate abdominal pain x3 days" |
| **S4** | Less Urgent | ESI-4 | Single-resource problem; stable. Minor injuries, simple infections, medication refill complications. | "mild ankle sprain", "sore throat 2 days" |
| **S5** | Non-Urgent | ESI-5 | No resources anticipated; routine/follow-up care, chronic stable conditions, administrative notes. | "routine follow-up", "annual physical", "refill request" |

**Design notes:**
- This is a 5-class *ordinal* label, not nominal — model evaluation must account for ordinality (e.g.
  report adjacent-level confusion separately from distant-level confusion; consider ordinal-aware losses
  as a later refinement, not in this data-layer pass).
- MTSamples is overwhelmingly composed of *retrospective documentation* (surgery notes, discharge summaries,
  SOAP notes), not real-time triage utterances. Most MTSamples documents will skew toward S3–S5 because
  they describe completed, often elective or routine, clinical encounters — this is an expected and
  important class imbalance, not a bug. The synthetic Hinglish patient-complaint layer (see
  `02_dataset_construction.md`) is what populates the S1/S2 end of the distribution, since MTSamples itself
  contains very few acute-presentation narratives written in first person.

## 3. Specialist routing: condensed department schema

MTSamples ships with 40 raw `medical_specialty` values, heavily imbalanced (Surgery=1103 down to
Allergy/Immunology=7) and containing categories that are not specialties at all (e.g. "SOAP / Chart /
Progress Notes", "Discharge Summary", "Office Notes", "Letters" are *document types*, not destinations).
For specialist-routing purposes we condense to **13 departments**, merging document-type artifacts into
their nearest clinical content category where determinable, and routing pure-administrative document
types to a catch-all.

| Department code | Department | Primary MTSamples specialties folded in |
|---|---|---|
| `ED` | Emergency Medicine | Emergency Room Reports |
| `CARDIO_PULM` | Cardiovascular & Pulmonary | Cardiovascular / Pulmonary, Sleep Medicine |
| `GI` | Gastroenterology | Gastroenterology, Bariatrics, Diets and Nutritions |
| `NEURO` | Neurology & Neurosurgery | Neurology, Neurosurgery |
| `ORTHO` | Orthopedics & Physical Medicine | Orthopedic, Physical Medicine - Rehab, Podiatry, Chiropractic |
| `SURGERY` | General & Specialty Surgery | Surgery, Cosmetic / Plastic Surgery |
| `OBGYN` | Obstetrics & Gynecology | Obstetrics / Gynecology |
| `PEDS` | Pediatrics | Pediatrics - Neonatal |
| `PSYCH` | Psychiatry & Mental Health | Psychiatry / Psychology |
| `ONCOLOGY_HEME` | Oncology & Hematology | Hematology - Oncology |
| `RENAL_URO` | Nephrology & Urology | Nephrology, Urology |
| `ENT_OPHTHALMO` | ENT, Ophthalmology & Dermatology | ENT - Otolaryngology, Ophthalmology, Dermatology, Allergy / Immunology |
| `GEN_MED` | General / Internal Medicine (catch-all) | General Medicine, Consult - History and Phy., SOAP/Chart/Progress Notes, Discharge Summary, Office Notes, Letters, Radiology, Endocrinology, Rheumatology, Pain Management, IME-QME-Work Comp, Lab Medicine - Pathology, Autopsy, Hospice - Palliative Care, Speech - Language, Dentistry |

**Design notes:**
- `GEN_MED` is intentionally a heterogeneous catch-all. It absorbs both genuinely general-medicine content
  and pure document-type categories (e.g. "Letters", "Autopsy") for which a content-based specialty cannot
  be inferred from the category label alone. We flag every row whose original label landed in this bucket
  via a `routing_confidence: low` field (see `data/processed/specialty_mapping.json`), so downstream
  consumers can filter or down-weight these rows rather than silently trusting them.
- This condensation is a documented, deterministic mapping (see `src/specialty_mapping.py`), not a
  model-learned one — it must be auditable and is the first thing a clinical reviewer should sanity-check.

## 4. Severity inference for MTSamples rows (heuristic, low-confidence)

MTSamples rows have **no ground-truth severity label**. We assign a provisional severity via a deterministic
regex/keyword scanner (`src/severity_heuristic.py`) operating on `transcription` text, as a *placeholder*
label to unblock pipeline development — explicitly **not** a validated clinical judgment. Rules:

1. Scan for S1 keyword set (arrest, code blue, unresponsive, exsanguinating, etc.) → if hit, label S1.
2. Else scan for S2 keyword set (severe + {pain, distress, bleeding}, stroke symptoms, acute MI language) → S2.
3. Else scan for S3 keyword set (persistent, moderate, recurrent + symptom terms) → S3.
4. Else scan for S5 keyword set (routine, follow-up, annual, refill, normal exam, no acute distress) → S5.
5. Default → S4 (the "stable minor problem" middle-of-the-road bucket), since most MTSamples documentation
   describes patients who are, by virtue of being coherently dictated post-hoc clinical notes, not in
   acute crisis at time of writing.

Every heuristically-labeled row carries `severity_label_source: "regex_heuristic_v0"` and
`severity_confidence: "low"` metadata fields. **This heuristic must not be treated as a training target of
record** — it exists to (a) sanity-check the taxonomy against real clinical text, (b) seed weak labels for
later semi-supervised refinement or clinician adjudication, and (c) stress-test the data pipeline end to
end. Any reported model metric trained against these labels alone must carry an explicit "heuristic-label,
unvalidated" caveat — this is a hygiene requirement for peer-review readiness, not optional.

## 5. Known limitations / what this taxonomy does NOT claim

- No inter-annotator agreement has been computed yet, because no human annotation has occurred — IAA
  methodology is specified for the *next* phase (clinician adjudication of a stratified sample) and is out
  of scope for this data-layer pass.
- The regex severity scanner will systematically misclassify negated findings ("no signs of distress"),
  historical mentions ("history of MI, now stable"), and family history ("mother had stroke") as more
  severe than the actual presentation, because it is keyword-based, not NLI/negation-aware. This is a known,
  documented failure mode, not an oversight (see the "history of cardiac arrest... now stable" regression
  case in `src/severity_heuristic.py`, which is deliberately kept as a documented S1 mislabel).
- **Empirically discovered during validation against the real MTSamples corpus** (not hypothetical): two
  clinical terms have a common *benign documentation sense* that collides with an *emergency sense*, and
  naive keyword matching cannot tell them apart without the fix applied below:
  - *"exsanguinated"* — in real operative notes this overwhelmingly refers to the routine surgical
    technique of draining a limb with an Esmarch bandage/tourniquet for a bloodless field, not hemorrhagic
    crisis. Fixed by requiring an explicit trauma/hemorrhage qualifier before treating it as S1 signal.
  - *"unresponsive"* — 33 of 57 (58%) of its occurrences in the raw corpus are the phrase "[condition]
    unresponsive to [therapy]" (conservative treatment failed, hence the procedure), not a patient in an
    unresponsive state. Fixed via negative lookahead excluding "unresponsive to ...".
  - Before these two fixes, the heuristic flagged 222 MTSamples rows as S1 (4.4% of corpus); after, 53
    rows (1.1%) — a 4x reduction in false positives from just two pattern refinements, on real data. This
    is direct empirical evidence for why this heuristic is explicitly labeled low-confidence: a clinician
    reviewer would catch these instantly, but they are exactly the kind of error a regex pass will make
    silently at scale without that scrutiny. Both cases are preserved as permanent regression tests.
- Severity distribution on the real corpus (after the above fixes): S1=53 (1.1%), S2=138 (2.8%),
  S3=111 (2.2%), S4=3994 (79.9%), S5=703 (14.1%). The heavy S4 skew is expected (§2) — MTSamples is
  retrospective documentation of largely non-acute encounters — but it also means the heuristic almost
  certainly still contains undiscovered false positives/negatives in the S1-S3 minority classes that
  haven't surfaced yet; the two fixes above are a lower bound on the real error rate, not a ceiling.
- The 13-department schema is a content-routing convenience, not a hospital org chart. Real specialist
  routing also depends on facility-specific resources, on-call schedules, and insurance/referral pathways
  that are entirely out of scope here.
- ESI itself, even when used by trained human triage nurses with real-time vital signs, mistriages up to
  roughly a third of patients in some studies — our text-only proxy task should be assumed *strictly*
  noisier than that baseline, not comparably accurate.

## 6. References (real, web-verified — no fabricated citations)

- Wuerz R, Milne LW, Eitel DR, Travers D, Gilboy N. ESI development and refinement; ENA-maintained.
- Validation of ESI v3 predicting ED resource consumption (retrospective cohort, n=403). *J Emerg Nurs* / PubMed PMID 14765078.
- ESI v5 simulation study on vital-sign-triggered uptriage at levels 3–5 (2025). *J Emerg Med* / *Acad Emerg Med* lineage.
- Emergency Severity Index Implementation Handbook, 5th ed. (AHRQ/ENA).
- JMIR Med Inform 2022;24(9):e37770 — telemedical query severity triage, 573 patient-generated queries
  from HealthTap/HealthcareMagic/iCliniq, transformer models (BERT/Bio+ClinicalBERT/SBERT) reaching ~0.90–0.92
  mean F1, substantially outperforming lexical/GloVe baselines. Directly motivates our transformer-based
  text severity classification approach.
- MT-Clinical BERT (PMC8449623) — shared-encoder multitask clinical information extraction with
  per-task heads; explicitly notes that naive loss-averaging across heads assumes constant inter-head loss
  scale, which does not generally hold — motivates the *weighted* joint loss design in this project rather
  than unweighted summation.
- MTSamples / Kaggle "Medical Transcriptions" dataset (Tara Boyle, scraped from mtsamples.com, CC0 Public
  Domain). Source of all real-clinical-text rows in this project.
- Hinglish/code-mixed NLP robustness literature: RCMT (arXiv:2403.16771) — robust perturbation-based joint
  training for code-mixed text via parameter sharing across clean/noisy word forms, directly informing our
  perturbation-engine design; hinglishNorm corpus motivating why Hinglish user-generated text requires
  explicit normalization/robustness handling; Hindi-English code-mixed sentiment analysis work using
  sub-word-level LSTM representations as precedent for robustness to non-canonical spelling.
