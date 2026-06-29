"""
build_dataset.py
-------------------
Orchestration script: runs the full data-layer pipeline end to end against
the raw MTSamples CSV and writes a processed, leakage-safe-split dataset to
data/processed/.

Pipeline stages:
  1. Load raw MTSamples CSV (data/raw/mtsamples.csv).
  2. Map each row's raw `medical_specialty` to one of the 13 condensed
     departments (src/specialty_mapping.py), recording routing_confidence.
  3. Score each row's severity via the deterministic regex heuristic
     (src/severity_heuristic.py), recording severity_label_source and
     severity_confidence metadata.
  4. For each seed row, generate the original (variant_index=0, unperturbed)
     row plus N Hinglish-perturbed variants (src/hinglish_perturbation.py),
     each carrying a unique, deterministic tracking_id
     (src/leakage_safe_split.py: make_tracking_id).
  5. Compute a leakage-safe train/val/test split at the SEED level, then
     propagate that split assignment to every derived row sharing a seed.
  6. Verify zero leakage before writing anything to disk -- this is a hard
     assertion, not a warning; the script aborts rather than persisting a
     leaking split.
  7. Write data/processed/dataset.csv (all derived rows + metadata columns)
     and data/processed/build_manifest.json (run parameters + summary stats,
     for reproducibility / audit trail).

This script does NOT call out to any LLM for text generation. The "Hinglish
variants" it produces are deterministic patient-style rewrites of the raw
provider note, followed by deterministic phonetic perturbations of those
Hinglish strings (a simple, auditable construction for this data-layer pass).
This is intentionally NOT the same as naive unconstrained LLM generation of
synthetic patient narratives, which is a separate, more elaborate pipeline
stage out of scope here (see docs/01_clinical_taxonomy.md framing on
heuristic vs validated labels). The point of this script is to validate that
every data-layer module composes correctly end-to-end against real data,
with tracking IDs and leakage-safe splitting working as designed -- not to
produce a publication-ready 15k-sample corpus in one pass.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from hinglish_perturbation import perturb_text  # noqa: E402
from leakage_safe_split import (  # noqa: E402
    assign_rows_to_split,
    compute_grouped_split,
    make_tracking_id,
    verify_no_leakage,
)
from severity_heuristic import score_severity  # noqa: E402
from specialty_mapping import map_specialty  # noqa: E402

RAW_CSV = REPO_ROOT / "data" / "raw" / "mtsamples.csv"
OUTPUT_CSV = REPO_ROOT / "data" / "processed" / "dataset.csv"
MANIFEST_JSON = REPO_ROOT / "data" / "processed" / "build_manifest.json"

# A small, illustrative bank of Hinglish patient-complaint framings used to
# steer each seed document into a first-person patient voice before
# perturbation. This is a deliberately minimal placeholder set for this
# data-layer integration pass, NOT the full ~15k-sample synthetic generation
# pipeline referenced in the project's broader scope -- see module docstring.
_HINGLISH_PREFIXES: tuple[str, ...] = (
    "Mera bahut dard ho raha hai, ",
    "Mujhe tabiyat ki shikayat hai, ",
    "Yeh dard bohot zyada hai kal subah se, ",
)

N_VARIANTS_PER_SEED = len(_HINGLISH_PREFIXES) + 1  # +1 for the unperturbed original

_NOTE_HEADER_PREFIXES: tuple[str, ...] = (
    "subjective:",
    "chief complaint:",
    "history of present illness:",
    "hpi:",
    "assessment and plan:",
    "plan:",
)


def _normalize_patient_voice(text: str) -> str:
    """
    Convert a provider note into a compact first-person Hinglish statement.

    The goal is not a literal translation. The goal is to remove the broken
    register switch caused by stitching a Hinglish greeting onto an English
    dictation note, while keeping the output deterministic and auditable.
    """
    cleaned = " ".join(str(text).split())
    lowered = cleaned.lower()
    for header in _NOTE_HEADER_PREFIXES:
        if lowered.startswith(header):
            cleaned = cleaned[len(header):].lstrip(" ,:-")
            lowered = cleaned.lower()
            break

    complaint = None
    complaint_patterns = (
        r"complaint of ([^.]+)",
        r"complains of ([^.]+)",
        r"with complaint of ([^.]+)",
        r"c/o ([^.]+)",
    )
    for pattern in complaint_patterns:
        match = re.search(pattern, cleaned, flags=re.IGNORECASE)
        if match and match.group(1).strip():
            complaint = match.group(1).strip().rstrip(".,;:")
            break

    age = None
    gender = None
    age_gender_match = re.search(r"(\d{1,3})[- ]year[- ]old\s+(male|female)", cleaned, flags=re.IGNORECASE)
    if age_gender_match:
        age = age_gender_match.group(1)
        gender = age_gender_match.group(2).lower()

    parts: list[str] = []
    if complaint:
        parts.append(f"Mujhe {complaint} ki shikayat hai")
    else:
        first_clause = cleaned.split(".", 1)[0].strip()
        if first_clause:
            parts.append(f"Mujhe {first_clause}")

    if age and gender:
        gender_word = "saal ki" if gender == "female" else "saal ka"
        parts.append(f"Main {age} {gender_word} hun")

    if not parts:
        parts.append("Mujhe tabiyat ki shikayat hai")

    return ". ".join(parts) + "."


def build_dataset(
    *,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    split_seed: int = 1337,
    perturbation_substitution_rate: float = 0.5,
    max_seed_rows: int | None = None,
) -> pd.DataFrame:
    """
    Run the full pipeline and return the resulting processed DataFrame
    (does not write to disk -- see main() for that).

    Parameters
    ----------
    max_seed_rows:
        If set, only process the first N seed rows from the raw CSV. Useful
        for fast local smoke runs; leave None to process the full corpus.
    """
    raw_df = pd.read_csv(RAW_CSV, index_col=0)
    raw_df["transcription"] = raw_df["transcription"].fillna("")
    if max_seed_rows is not None:
        raw_df = raw_df.head(max_seed_rows)

    seed_ids = [str(idx) for idx in raw_df.index]

    split_result = compute_grouped_split(
        seed_ids,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        random_seed=split_seed,
    )

    rows: list[dict] = []
    for idx, raw_row in raw_df.iterrows():
        seed_id = str(idx)
        transcription = raw_row["transcription"]
        department_code, routing_confidence = map_specialty(raw_row["medical_specialty"])

        # Variant 0: original, unperturbed English transcription.
        original_severity = score_severity(transcription)
        rows.append(
            {
                "tracking_id": make_tracking_id(seed_id, 0),
                "seed_id": seed_id,
                "variant_index": 0,
                "is_perturbed": False,
                "language": "en",
                "text": transcription,
                "raw_medical_specialty": raw_row["medical_specialty"].strip(),
                "department_code": department_code,
                "routing_confidence": routing_confidence,
                "severity_heuristic": original_severity.severity,
                "severity_label_source": original_severity.label_source,
                "severity_confidence": original_severity.confidence,
            }
        )

        # Variants 1..N: patient-style Hinglish rewrites, then perturbed.
        for variant_idx, prefix in enumerate(_HINGLISH_PREFIXES, start=1):
            patient_voice = _normalize_patient_voice(transcription)
            combined_text = f"{prefix}{patient_voice}"
            # Deterministic seed for perturbation derived from the tracking
            # ID's hash component so it's stable across rebuilds without
            # depending on global random state or wall-clock time.
            tracking_id = make_tracking_id(seed_id, variant_idx)
            perturbation_seed = int(tracking_id.split("::")[-1], 16)
            perturbation_result = perturb_text(
                combined_text,
                seed=perturbation_seed,
                substitution_rate=perturbation_substitution_rate,
            )
            variant_severity = score_severity(perturbation_result.perturbed)
            rows.append(
                {
                    "tracking_id": tracking_id,
                    "seed_id": seed_id,
                    "variant_index": variant_idx,
                    "is_perturbed": True,
                    "language": "hinglish",
                    "text": perturbation_result.perturbed,
                    "raw_medical_specialty": raw_row["medical_specialty"].strip(),
                    "department_code": department_code,
                    "routing_confidence": routing_confidence,
                    "severity_heuristic": variant_severity.severity,
                    "severity_label_source": variant_severity.label_source,
                    "severity_confidence": variant_severity.confidence,
                }
            )

    dataset_df = pd.DataFrame(rows)

    split_assignments = assign_rows_to_split(
        dataset_df["tracking_id"].tolist(), split_result
    )
    dataset_df["split"] = dataset_df["tracking_id"].map(split_assignments)

    # Hard assertion: abort rather than persist a leaking split.
    by_split = {
        split_name: dataset_df.loc[dataset_df["split"] == split_name, "tracking_id"].tolist()
        for split_name in ("train", "val", "test")
    }
    if not verify_no_leakage(by_split):
        raise RuntimeError(
            "LEAKAGE DETECTED in computed split -- aborting before writing "
            "any output. This indicates a bug in the split/assignment logic, "
            "not a data problem; do not bypass this check."
        )

    return dataset_df


def main() -> None:
    print("Building MediTriageAI processed dataset from raw MTSamples...")
    dataset_df = build_dataset()

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    dataset_df.to_csv(OUTPUT_CSV, index=False)

    n_seed_docs = dataset_df["seed_id"].nunique()
    split_seed_counts = (
        dataset_df.drop_duplicates("seed_id")["split"].value_counts().to_dict()
    )
    split_row_counts = dataset_df["split"].value_counts().to_dict()
    severity_counts = dataset_df["severity_heuristic"].value_counts().sort_index().to_dict()
    department_counts = dataset_df["department_code"].value_counts().to_dict()
    routing_confidence_counts = dataset_df["routing_confidence"].value_counts().to_dict()

    manifest = {
        "build_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_csv": str(RAW_CSV.relative_to(REPO_ROOT)),
        "n_seed_documents": n_seed_docs,
        "n_variants_per_seed": N_VARIANTS_PER_SEED,
        "n_total_rows": len(dataset_df),
        "split_seed_counts": split_seed_counts,
        "split_row_counts": split_row_counts,
        "severity_heuristic_distribution": severity_counts,
        "department_distribution": department_counts,
        "routing_confidence_distribution": routing_confidence_counts,
        "leakage_verified": True,
        "caveats": [
            "severity_heuristic is a deterministic regex heuristic, NOT a "
            "validated clinical label -- see docs/01_clinical_taxonomy.md "
            "Section 4-5.",
            "Hinglish variants are deterministic phonetic perturbations of a "
            "small illustrative prefix bank, NOT the full synthetic "
            "generation pipeline -- see build_dataset.py module docstring.",
            "routing_confidence='low' rows (GEN_MED catch-all from "
            "document-type-artifact specialties) should be filtered or "
            "down-weighted by downstream consumers needing high-confidence "
            "specialty labels.",
        ],
    }
    with MANIFEST_JSON.open("w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nWrote {len(dataset_df)} rows ({n_seed_docs} seed docs x "
          f"{N_VARIANTS_PER_SEED} variants) to {OUTPUT_CSV.relative_to(REPO_ROOT)}")
    print(f"Seed-level split: {split_seed_counts}")
    print(f"Row-level split:  {split_row_counts}")
    print(f"Severity heuristic distribution: {severity_counts}")
    print(f"Manifest written to {MANIFEST_JSON.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
