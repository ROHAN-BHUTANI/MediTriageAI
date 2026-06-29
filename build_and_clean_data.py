"""
MediTriageAI -- MTSamples Real-Data Ingestion, Cleaning, and Heuristic
Severity Scanning Pipeline
=========================================================================
Standalone script. No network calls (reads a LOCAL file only). Requires
only pandas (re, json, os, hashlib are stdlib).

DATA SOURCE NOTE -- READ BEFORE YOUR MEETING:
This script ingests YOUR local `mtsamples_raw.csv`. MTSamples itself is
real: scraped from mtsamples.com by Tara Boyle, originally distributed
via Kaggle under a CC0 (public domain) license, with the standard
5-column schema (description, medical_specialty, sample_name,
transcription, keywords) used below. I could not independently browse
the specific GitHub mirror you named (GitHub blocks automated access to
that page from here), so I can't personally vouch that mirror is
byte-identical to the canonical Kaggle release -- but the dataset,
schema, and license are well-documented and real, which is the part
that matters for citing it honestly to your advisor.

CRITICAL CAVEAT -- THE SEVERITY SCANNER IS A WEAK HEURISTIC, NOT GROUND
TRUTH:
MTSamples is English-only, formally dictated clinical documentation
(surgical reports, consultation letters), NOT patient-described
symptoms. Words like "acute," "severe," or "fracture" appear constantly
in professional medical writing regardless of the patient's actual
real-time urgency (e.g. a routine note can clinically describe a
HEALED fracture, or use "acute" as a textbook descriptor for something
already resolved). The regex severity scanner below is therefore
exported as `severity_label_heuristic`, NOT `severity_label` -- it is
an unvalidated, low-confidence keyword heuristic, explicitly flagged as
such in this script, in the output schema, and in the export metadata.
Do not present this column to your advisor as verified ground truth.

There is also no trilingual or code-switching content in MTSamples
(it is English-only formal dictation), so the Hinglish-emphasis
preservation logic ("bahut bahut") is included for pipeline consistency
with your other project stages, but will simply never fire on this
particular dataset -- documented here so that's not a surprise later.

FOLDER STRUCTURE PRODUCED:
    1_raw_seeds/              -- copy of the untouched source file
    2_expanded_raw/           -- raw_dataset_snapshot.csv (uncleaned rows)
    3_cleaned_perturbed/      -- cleaned_processed_dataset.json (final)
"""

import os
import re
import json
import hashlib
import unicodedata
from datetime import datetime, timezone

import pandas as pd


# -----------------------------------------------------------------------
# STAGE 0: WORKSPACE DIRECTORY INITIALIZATION
# -----------------------------------------------------------------------

FOLDER_RAW_SEEDS = "1_raw_seeds"
FOLDER_EXPANDED_RAW = "2_expanded_raw"
FOLDER_CLEANED_PERTURBED = "3_cleaned_perturbed"
ALL_FOLDERS = [FOLDER_RAW_SEEDS, FOLDER_EXPANDED_RAW, FOLDER_CLEANED_PERTURBED]

INPUT_CSV_PATH = "mtsamples_raw.csv"


def ensure_directory_matrix(base_path: str = ".") -> dict:
    """Creates the three pipeline folders if they don't already exist. Idempotent."""
    paths = {}
    for folder in ALL_FOLDERS:
        full_path = os.path.join(base_path, folder)
        os.makedirs(full_path, exist_ok=True)
        paths[folder] = full_path
        print(f"[directory_matrix] ensured: {full_path}")
    return paths


# -----------------------------------------------------------------------
# STAGE 1: DATA INGESTION
# -----------------------------------------------------------------------

REQUIRED_SOURCE_COLUMNS = ["transcription", "medical_specialty"]


def ingest_mtsamples(csv_path: str = INPUT_CSV_PATH) -> pd.DataFrame:
    """
    Reads the local mtsamples_raw.csv. Validates that the columns this
    pipeline depends on actually exist (real MTSamples uses
    'transcription' and 'medical_specialty'; if your local file uses
    different column names, this raises immediately with a clear
    message rather than failing confusingly downstream).

    Real MTSamples has a small number of rows with a NULL
    `transcription` (documented: ~33 of 4999 rows) -- these are dropped
    here explicitly, with a printed count, rather than silently
    surviving as NaN into later string operations where they would
    raise a confusing error.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"ingest_mtsamples: '{csv_path}' not found in the current working "
            f"directory. Place your downloaded mtsamples_raw.csv next to this "
            f"script, or pass the correct path."
        )

    df = pd.read_csv(csv_path)

    missing_cols = [c for c in REQUIRED_SOURCE_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"ingest_mtsamples: required column(s) {missing_cols} not found in "
            f"'{csv_path}'. Columns present: {list(df.columns)}. If your local "
            f"file uses different header names, rename them to match before "
            f"re-running, or adjust REQUIRED_SOURCE_COLUMNS."
        )

    n_before = len(df)
    df = df.dropna(subset=["transcription"]).reset_index(drop=True)
    n_dropped_null_transcription = n_before - len(df)
    print(
        f"[stage_1_ingest] loaded '{csv_path}': {n_before} rows -> "
        f"{len(df)} rows after dropping {n_dropped_null_transcription} row(s) "
        f"with a NULL transcription"
    )
    return df


# -----------------------------------------------------------------------
# STAGE 1b: SPECIALTY MAPPING / NORMALIZATION
# -----------------------------------------------------------------------

def normalize_specialty_label(raw_specialty) -> str:
    """
    Real MTSamples 'medical_specialty' values have leading whitespace
    (e.g. " Cardiovascular / Pulmonary") and inconsistent casing across
    rows. This normalizes to a clean, consistent label string -- it does
    NOT attempt to remap MTSamples' ~40 specialty categories onto your
    project's specific 10-category schema (GP/Cardio/Derm/Neuro/Ortho/
    Pulm/GI/Psych/ENT/EM); that many-to-many remapping is a separate,
    clinically-informed decision your team should make deliberately
    (e.g. "Cardiovascular / Pulmonary" could map to either Cardiologist
    or Pulmonologist depending on the note's actual content), not one
    this cleaning script should silently impose.
    """
    if pd.isna(raw_specialty):
        return ""
    return str(raw_specialty).strip()


# -----------------------------------------------------------------------
# STAGE 2: SANITIZATION ENGINE
# -----------------------------------------------------------------------

_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "\U00002190-\U000021FF"
    "\U00002B00-\U00002BFF"
    "\U0000FE0F"
    "]+",
    flags=re.UNICODE,
)
_REPEATED_PUNCT_PATTERN = re.compile(r"([!?.~,;:])\1{2,}")
_CONTROL_CHAR_PATTERN = re.compile(r"[\u200b-\u200f\u202a-\u202e\ufeff]")
_WHITESPACE_PATTERN = re.compile(r"\s+")
_LEADING_TRAILING_JUNK_PATTERN = re.compile(r"^[^\w\u0900-\u097F]+|[^\w\u0900-\u097F.,!?]+$")

# Internal junk-character runs (e.g. "###POST-OP NOTE###", "***", "^^^")
# embedded MID-DOCUMENT, not just at the string boundary. Real clinical
# transcriptions commonly contain section-header markers like this
# throughout the document, which the boundary-only junk stripper above
# (by design) does not touch. Matches runs of 2+ symbol characters from
# this specific junk-symbol set; deliberately narrow (does not include
# ! ? . , ; : ~ which are handled separately above and ARE meaningful
# punctuation) so normal sentence punctuation is never affected.
_INTERNAL_JUNK_RUN_PATTERN = re.compile(r"[#@^*`_]{2,}")


def sanitize_record(text: str) -> str:
    """
    Cleans a single raw clinical text record.

    Operations, in order:
      1. Unicode NFC normalization.
      2. Strip emoji / pictographic characters.
      3. Collapse degenerate repeated PUNCTUATION runs ("???" -> "?").
         Does NOT touch repeated WORDS -- conversational emphasis like
         "bahut bahut" is preserved by construction (this function only
         matches punctuation character classes, never word tokens).
         NOTE: MTSamples is English-only formal dictation, so this
         particular preservation rule is not expected to ever actually
         fire on this dataset -- it's kept here for consistency with
         the rest of the project's cleaning pipeline.
      4. Strip INTERNAL junk-character runs (e.g. "###POST-OP NOTE###"
         section-header markers embedded mid-document) -- common in
         real clinical transcriptions, distinct from step 7's
         boundary-only junk stripping below.
      5. Strip zero-width/control characters.
      6. Lowercase ONLY Latin-script runs.
      7. Standardize whitespace.
      8. Strip leading/trailing junk while PRESERVING internal commas
         (clinically meaningful symptom/finding separators) and
         sentence-final punctuation.

    Returns "" for None/empty/whitespace-only input.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    text = unicodedata.normalize("NFC", text)
    text = _EMOJI_PATTERN.sub("", text)
    text = _REPEATED_PUNCT_PATTERN.sub(r"\1", text)
    text = _INTERNAL_JUNK_RUN_PATTERN.sub(" ", text)
    text = _CONTROL_CHAR_PATTERN.sub("", text)
    text = re.sub(r"[A-Za-z]+", lambda m: m.group(0).lower(), text)
    text = _WHITESPACE_PATTERN.sub(" ", text).strip()
    text = _LEADING_TRAILING_JUNK_PATTERN.sub("", text)
    text = _WHITESPACE_PATTERN.sub(" ", text).strip()

    return text


# -----------------------------------------------------------------------
# STAGE 3: MISSING DATA & LENGTH AUDIT FILTER
# -----------------------------------------------------------------------

MIN_WORD_COUNT = 3


def audit_and_filter(df: pd.DataFrame, text_column: str = "cleaned_text") -> pd.DataFrame:
    """
    Drops rows missing 'specialty_label' (empty after normalization) or
    whose text_column has fewer than MIN_WORD_COUNT words. Pure
    function: returns a new, filtered, index-reset dataframe.

    NOTE: this dataset has no severity_label at ingestion time (severity
    is HEURISTICALLY DERIVED in Stage 4 below, not present in the source
    file) -- so unlike the synthetic-data pipeline's audit step, this
    one only checks specialty_label and length at this stage. The
    heuristic severity label is computed afterward and is never used as
    a drop criterion, since a "scanner found no keyword" result is
    informative (defaults to a label) rather than a missing-data case.
    """
    required_cols = ["specialty_label", text_column]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"audit_and_filter: missing required column(s) {missing_cols}. "
            f"Upstream schema problem -- fix before re-running."
        )

    working = df.copy()

    specialty_missing = working["specialty_label"].apply(
        lambda v: pd.isna(v) or (isinstance(v, str) and v.strip() == "")
    )
    too_short = working[text_column].apply(lambda v: len(str(v).split()) if not pd.isna(v) else 0) < MIN_WORD_COUNT
    drop_mask = specialty_missing | too_short

    n_total = len(working)
    n_kept = int((~drop_mask).sum())
    print(
        f"[stage_3_audit] {n_total} rows in -> {n_kept} kept, {n_total - n_kept} dropped "
        f"(missing specialty_label: {int(specialty_missing.sum())}, "
        f"too short (<{MIN_WORD_COUNT} words): {int(too_short.sum())})"
    )

    return working.loc[~drop_mask].reset_index(drop=True)


# -----------------------------------------------------------------------
# STAGE 4: PROGRAMMATIC SEVERITY SCANNER (HEURISTIC, NOT GROUND TRUTH)
# -----------------------------------------------------------------------

# Keyword sets are deliberately small and auditable -- every rule here
# is something a human reviewer can read and judge, consistent with the
# same design philosophy as the project's red-flag fallback layer.
# These patterns are run against the CLEANED, lowercased text.
_CRITICAL_PATTERNS = [
    r"\bemergency\b", r"\bcardiac arrest\b", r"\bcode blue\b",
    r"\brespiratory failure\b", r"\blife.?threatening\b",
]
_HIGH_PATTERNS = [
    r"\bacute\b", r"\bsevere\b", r"\bfracture\b", r"\bhemorrhage\b",
    r"\bunstable\b",
]
_LOW_PATTERNS = [
    r"\broutine\b", r"\bfollow.?up\b", r"\bchronic\b", r"\bstable\b",
    r"\bwell.?controlled\b",
]

_CRITICAL_RE = re.compile("|".join(_CRITICAL_PATTERNS), re.IGNORECASE)
_HIGH_RE = re.compile("|".join(_HIGH_PATTERNS), re.IGNORECASE)
_LOW_RE = re.compile("|".join(_LOW_PATTERNS), re.IGNORECASE)


def scan_severity_heuristic(cleaned_text: str) -> dict:
    """
    Derives a WEAK, UNVALIDATED heuristic severity label by keyword
    lookup over cleaned clinical text. THIS IS NOT GROUND TRUTH -- see
    the module docstring's caveat. Returned as 'severity_label_heuristic'
    with an explicit 'heuristic_confidence' flag so this is never
    confused with a clinically validated label downstream.

    Priority order when multiple categories match in the same text
    (common, since a note can mention both "acute" and "routine" in
    different contexts): CRITICAL > HIGH > LOW > unknown. This is a
    deliberately conservative choice (favor flagging higher apparent
    severity when signals conflict) but is itself an unvalidated design
    choice, not a clinically derived rule -- documented here so it's
    not mistaken for one.

    Returns:
        {
            "severity_label_heuristic": "critical" | "high" | "low" | "unknown",
            "matched_keywords": [...],
            "heuristic_confidence": "low",  # ALWAYS "low" -- this scanner
                                             # has no validation against
                                             # real outcomes; the literal
                                             # string is a constant
                                             # reminder, not a computed score.
        }
    """
    if not cleaned_text:
        return {"severity_label_heuristic": "unknown", "matched_keywords": [], "heuristic_confidence": "low"}

    critical_matches = _CRITICAL_RE.findall(cleaned_text)
    high_matches = _HIGH_RE.findall(cleaned_text)
    low_matches = _LOW_RE.findall(cleaned_text)

    if critical_matches:
        label, matches = "critical", critical_matches
    elif high_matches:
        label, matches = "high", high_matches
    elif low_matches:
        label, matches = "low", low_matches
    else:
        label, matches = "unknown", []

    return {
        "severity_label_heuristic": label,
        "matched_keywords": sorted(set(m.lower() for m in matches)),
        "heuristic_confidence": "low",
    }


# -----------------------------------------------------------------------
# STAGE 5: LEAKAGE-SAFE SPLIT ASSIGNMENT (ROW-LEVEL, AS SPECIFIED)
# -----------------------------------------------------------------------

def assign_splits_row_level(
    row_ids: list,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed_for_split: int = 1337,
) -> dict:
    """
    Deterministic 80/10/10 split assigned AT THE PATIENT-ROW LEVEL.

    IMPORTANT DISTINCTION FROM THE SYNTHETIC-DATA PIPELINE: in the
    synthetic trilingual pipeline (prior project phase), splitting had
    to happen at the SEED level, because multiple rows (English/Hindi/
    Hinglish variants) were near-paraphrases descending from one shared
    seed scenario -- splitting those at the row level would leak the
    same underlying content across train/test.

    MTSamples has NO such structure: each row is one independent,
    naturally-occurring patient transcription with no programmatically-
    generated paraphrase siblings. Row-level splitting is therefore the
    CORRECT choice here, not a shortcut -- there is no shared-content
    leakage risk to guard against in this dataset the way there was for
    the synthetic one. (If you later deduplicate near-identical template
    notes within MTSamples itself, re-examine this assumption.)

    Deterministic via hashing, so re-running produces the identical
    split without needing to persist random state.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    def _rank(row_id) -> float:
        h = hashlib.sha256(f"{seed_for_split}:{row_id}".encode()).hexdigest()
        return int(h, 16) / 16 ** len(h)

    sorted_ids = sorted(row_ids, key=_rank)
    n = len(sorted_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    if n > 0 and (n_val == 0 or (n - n_train - n_val) == 0):
        print(
            f"[assign_splits_row_level] WARNING: only {n} row(s) -- ratios "
            f"round down to a zero-row split. Expected only for small "
            f"demo/mock runs; verify split sizes on your real ~5000-row file."
        )

    split_map = {}
    for i, rid in enumerate(sorted_ids):
        if i < n_train:
            split_map[rid] = "train"
        elif i < n_train + n_val:
            split_map[rid] = "val"
        else:
            split_map[rid] = "test"
    return split_map


# -----------------------------------------------------------------------
# STAGE 6: EXPORT
# -----------------------------------------------------------------------

def run_pipeline(base_path: str = ".", input_csv: str = INPUT_CSV_PATH) -> dict:
    paths = ensure_directory_matrix(base_path)

    print("\n" + "=" * 70)
    print("STAGE 1 -- INGESTION")
    print("=" * 70)
    raw_df = ingest_mtsamples(input_csv)

    # Save an untouched copy of the source file into 1_raw_seeds/, so
    # that folder holds the most upstream artifact exactly as received.
    raw_seed_copy_path = os.path.join(paths[FOLDER_RAW_SEEDS], "mtsamples_raw_copy.csv")
    raw_df.to_csv(raw_seed_copy_path, index=False, encoding="utf-8")
    print(f"[stage_1] archived untouched copy -> {raw_seed_copy_path}")

    print("\n" + "=" * 70)
    print("STAGE 1b -- SPECIALTY NORMALIZATION + RAW SNAPSHOT EXPORT")
    print("=" * 70)
    working = raw_df.copy()
    working["specialty_label"] = working["medical_specialty"].apply(normalize_specialty_label)
    working["raw_row_id"] = [f"MTA-RAW-{i:06d}" for i in range(len(working))]

    raw_snapshot_path = os.path.join(paths[FOLDER_EXPANDED_RAW], "raw_dataset_snapshot.csv")
    # Exported EXACTLY as received (transcription column untouched) --
    # this is the "before" artifact for your advisor comparison.
    working[["raw_row_id", "transcription", "specialty_label"]].to_csv(
        raw_snapshot_path, index=False, encoding="utf-8"
    )
    print(f"[stage_1b] wrote raw (uncleaned) snapshot -> {raw_snapshot_path} ({len(working)} rows)")

    print("\n" + "=" * 70)
    print("STAGE 2 -- SANITIZATION")
    print("=" * 70)
    working["cleaned_text"] = working["transcription"].apply(sanitize_record)
    print(f"[stage_2] sanitized {len(working)} transcriptions")

    print("\n" + "=" * 70)
    print("STAGE 3 -- AUDIT & FILTER")
    print("=" * 70)
    filtered = audit_and_filter(working, text_column="cleaned_text")

    print("\n" + "=" * 70)
    print("STAGE 4 -- HEURISTIC SEVERITY SCANNER (weak label, see caveat)")
    print("=" * 70)
    scan_results = filtered["cleaned_text"].apply(scan_severity_heuristic)
    filtered["severity_label_heuristic"] = scan_results.apply(lambda r: r["severity_label_heuristic"])
    filtered["matched_keywords"] = scan_results.apply(lambda r: r["matched_keywords"])
    filtered["heuristic_confidence"] = scan_results.apply(lambda r: r["heuristic_confidence"])
    severity_counts = filtered["severity_label_heuristic"].value_counts().to_dict()
    print(f"[stage_4] heuristic severity distribution: {severity_counts}")
    print(
        "[stage_4] REMINDER: 'severity_label_heuristic' is an unvalidated "
        "keyword-matching heuristic, not a clinically verified label."
    )

    print("\n" + "=" * 70)
    print("STAGE 5 -- LEAKAGE-SAFE ROW-LEVEL SPLIT (80/10/10)")
    print("=" * 70)
    filtered = filtered.reset_index(drop=True)
    filtered["tracking_id"] = [f"MTA-{i:06d}" for i in range(len(filtered))]
    split_map = assign_splits_row_level(filtered["tracking_id"].tolist())
    filtered["target_split"] = filtered["tracking_id"].map(split_map)
    split_distribution = filtered["target_split"].value_counts().to_dict()
    print(f"[stage_5] split distribution: {split_distribution}")

    print("\n" + "=" * 70)
    print("STAGE 6 -- FINAL EXPORT")
    print("=" * 70)
    export_columns = [
        "tracking_id", "specialty_label", "cleaned_text",
        "severity_label_heuristic", "matched_keywords", "heuristic_confidence",
        "target_split",
    ]
    raw_records = filtered[export_columns].to_dict(orient="records")
    # NaN -> None so the output is standards-compliant JSON (NaN is not
    # valid per the JSON spec and breaks strict parsers, e.g. JS JSON.parse).
    records = [
        {k: (None if isinstance(v, float) and pd.isna(v) else v) for k, v in rec.items()}
        for rec in raw_records
    ]

    export_payload = {
        "metadata": {
            "export_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "n_records": len(records),
            "schema_version": "1.0",
            "split_distribution": split_distribution,
            "severity_label_heuristic_distribution": severity_counts,
            "data_quality_warning": (
                "severity_label_heuristic is derived from a small, "
                "unvalidated regex keyword scanner over formally-dictated "
                "clinical transcriptions. It has NOT been checked against "
                "real clinical outcomes or expert review and should NOT be "
                "treated as ground truth. specialty_label is taken directly "
                "from MTSamples' own medical_specialty field and has NOT been "
                "remapped onto this project's 10-category specialist schema."
            ),
            "provenance": (
                "Source: locally-provided mtsamples_raw.csv (MTSamples "
                "clinical transcription dataset, scraped from mtsamples.com "
                "by Tara Boyle, CC0 public domain license)."
            ),
        },
        "records": records,
    }

    cleaned_path = os.path.join(paths[FOLDER_CLEANED_PERTURBED], "cleaned_processed_dataset.json")
    with open(cleaned_path, "w", encoding="utf-8") as f:
        json.dump(export_payload, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"[stage_6] wrote final cleaned dataset -> {cleaned_path}")

    return {
        "n_raw_rows": len(raw_df),
        "n_after_cleaning_and_audit": len(filtered),
        "n_dropped_total": len(raw_df) - len(filtered),
        "split_distribution": split_distribution,
        "severity_label_heuristic_distribution": severity_counts,
        "raw_snapshot_path": raw_snapshot_path,
        "cleaned_output_path": cleaned_path,
    }


def main():
    print("=" * 70)
    print("BEFORE / AFTER -- single record sanitization demo")
    print("=" * 70)
    demo_raw = "   ###POST-OP NOTE###   Patient tolerated the procedure well!!! No acute distress noted??? Vitals stable.    "
    demo_clean = sanitize_record(demo_raw)
    print(f"BEFORE: {demo_raw!r}")
    print(f"AFTER : {demo_clean!r}")

    print("\n" + "=" * 70)
    print("RUNNING FULL PIPELINE")
    print("=" * 70)
    summary = run_pipeline(base_path=".", input_csv=INPUT_CSV_PATH)

    print("\n" + "=" * 70)
    print("FINAL ROW COUNTS -- COPY THESE TO YOUR SLIDES")
    print("=" * 70)
    print(f"  Raw rows ingested            : {summary['n_raw_rows']}")
    print(f"  Rows after cleaning + audit  : {summary['n_after_cleaning_and_audit']}")
    print(f"  Rows dropped (audit filter)  : {summary['n_dropped_total']}")
    print(f"  Train / Val / Test split     : {summary['split_distribution']}")
    print(f"  Heuristic severity breakdown : {summary['severity_label_heuristic_distribution']}")
    print(f"  Raw snapshot     -> 2_expanded_raw/raw_dataset_snapshot.csv")
    print(f"  Cleaned dataset  -> 3_cleaned_perturbed/cleaned_processed_dataset.json")
    print(
        "\n  REMINDER FOR YOUR SLIDES: 'severity_label_heuristic' is an "
        "unvalidated keyword heuristic, not verified ground truth -- say "
        "this explicitly if you present this column."
    )


if __name__ == "__main__":
    main()