"""
test_data_layer_integration.py
---------------------------------
End-to-end integration test for the MediTriageAI data layer. Exercises:
  1. specialty_mapping against the real downloaded MTSamples CSV
  2. severity_heuristic against the real corpus + documented regression cases
  3. hinglish_perturbation determinism guarantees
  4. leakage_safe_split correctness + the small-N guard

Run with: python3 -m pytest tests/test_data_layer_integration.py -v
(or just: python3 tests/test_data_layer_integration.py for a plain run)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from hinglish_perturbation import is_deterministic, perturb_text  # noqa: E402
from leakage_safe_split import (  # noqa: E402
    InsufficientSeedsError,
    assign_rows_to_split,
    compute_grouped_split,
    make_tracking_id,
    verify_no_leakage,
)
from severity_heuristic import score_severity  # noqa: E402
from specialty_mapping import all_raw_labels_mapped, map_specialty  # noqa: E402

MTSAMPLES_CSV = REPO_ROOT / "data" / "raw" / "mtsamples.csv"


def _load_mtsamples() -> pd.DataFrame:
    df = pd.read_csv(MTSAMPLES_CSV, index_col=0)
    df["transcription"] = df["transcription"].fillna("")
    return df


def test_mtsamples_csv_present_and_well_formed() -> None:
    assert MTSAMPLES_CSV.exists(), f"Expected MTSamples CSV at {MTSAMPLES_CSV}"
    df = _load_mtsamples()
    assert len(df) == 4999, f"Expected 4999 rows, got {len(df)}"
    expected_cols = {"description", "medical_specialty", "sample_name", "transcription", "keywords"}
    assert expected_cols.issubset(set(df.columns))


def test_specialty_mapping_covers_all_real_raw_labels() -> None:
    df = _load_mtsamples()
    distinct_labels = df["medical_specialty"].unique().tolist()
    assert len(distinct_labels) == 40, f"Expected 40 raw specialty labels, got {len(distinct_labels)}"
    assert all_raw_labels_mapped(distinct_labels), (
        "Schema drift detected: some raw medical_specialty value in the real "
        "CSV has no entry in RAW_TO_DEPARTMENT. Update src/specialty_mapping.py."
    )


def test_specialty_mapping_produces_exactly_13_departments() -> None:
    df = _load_mtsamples()
    departments_seen = {map_specialty(label)[0] for label in df["medical_specialty"]}
    assert len(departments_seen) == 13, f"Expected 13 departments, got {len(departments_seen)}"


def test_specialty_mapping_unknown_label_raises() -> None:
    try:
        map_specialty("Definitely Not A Real Specialty XYZ")
        raise AssertionError("Expected KeyError for unrecognized specialty, none raised")
    except KeyError:
        pass  # expected


def test_severity_heuristic_regression_cases() -> None:
    """The documented false-positive fixes must hold permanently."""
    cases = [
        ("The foot was elevated and exsanguinated with an Esmarch bandage "
         "prior to inflation of the pneumatic tourniquet.", "S4"),
        ("Patient exsanguinated rapidly secondary to traumatic gunshot wound.", "S1"),
        ("Severe menometrorrhagia unresponsive to medical therapy, "
         "proceeding to hysterectomy.", "S4"),
        ("Patient found unresponsive at home by family, EMS called.", "S1"),
    ]
    for text, expected in cases:
        result = score_severity(text)
        assert result.severity == expected, (
            f"Regression failure: expected {expected}, got {result.severity} "
            f"for text={text!r}"
        )
        assert result.label_source == "regex_heuristic_v0"
        assert result.confidence == "low"


def test_severity_heuristic_real_corpus_distribution_sane() -> None:
    """
    Sanity bound, not a precise expectation: after the documented false-
    positive fixes, S1 should be a small minority of the corpus (MTSamples is
    mostly retrospective non-acute documentation per docs/01_clinical_taxonomy.md).
    This is a regression guard against someone re-introducing a broad S1
    pattern that balloons false positives again.
    """
    df = _load_mtsamples()
    results = df["transcription"].apply(score_severity)
    severities = results.apply(lambda r: r.severity)
    s1_fraction = (severities == "S1").mean()
    assert s1_fraction < 0.02, (
        f"S1 fraction {s1_fraction:.3f} is suspiciously high -- check for a "
        f"regression in the exsanguination/unresponsive-to-therapy fixes."
    )


def test_hinglish_perturbation_is_deterministic() -> None:
    samples = [
        "Mera bahut dard ho raha hai, kal subah se theek nahi hun.",
        "Yeh dard bohot zyada hai, raat ko sone nahi de raha.",
    ]
    for text in samples:
        for seed in (1, 42, 12345):
            assert is_deterministic(text, seed), (
                f"Non-deterministic output for text={text!r} seed={seed}"
            )


def test_hinglish_perturbation_different_seeds_can_differ() -> None:
    text = "Mera bahut dard ho raha hai, kal subah se theek nahi hun."
    outputs = {perturb_text(text, seed=s, substitution_rate=0.6).perturbed for s in range(10)}
    assert len(outputs) > 1, "Expected variation across different seeds, got identical output"


def test_hinglish_perturbation_zero_rate_is_noop() -> None:
    text = "Mera bahut dard ho raha hai, kal subah se theek nahi hun."
    result = perturb_text(text, seed=1, substitution_rate=0.0)
    assert result.perturbed == text
    assert result.substitutions_applied == []


def test_leakage_safe_split_normal_case() -> None:
    seed_ids = [f"seed_{i}" for i in range(200)]
    result = compute_grouped_split(seed_ids, random_seed=7)
    assert sum(result.seed_counts.values()) == 200
    assert result.seed_counts["train"] > 0
    assert result.seed_counts["val"] > 0
    assert result.seed_counts["test"] > 0


def test_leakage_safe_split_small_n_guard_raises() -> None:
    tiny = [f"seed_{i}" for i in range(5)]
    try:
        compute_grouped_split(tiny, train_fraction=0.8, val_fraction=0.1, test_fraction=0.1)
        raise AssertionError("Expected InsufficientSeedsError, none raised")
    except InsufficientSeedsError:
        pass  # expected


def test_leakage_safe_split_no_leakage_with_multiple_variants_per_seed() -> None:
    seed_ids = [f"seed_{i}" for i in range(50)]
    split_result = compute_grouped_split(seed_ids, random_seed=99)

    tracking_ids = [
        make_tracking_id(sid, variant_idx)
        for sid in seed_ids
        for variant_idx in range(4)  # 1 original + 3 perturbed variants
    ]
    row_assignments = assign_rows_to_split(tracking_ids, split_result)

    by_split: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for tid, split in row_assignments.items():
        by_split[split].append(tid)

    assert verify_no_leakage(by_split), "Leakage detected in grouped split!"
    # every seed's 4 variants must be entirely within one split
    assert sum(len(v) for v in by_split.values()) == 200


def test_verify_no_leakage_catches_deliberate_leak() -> None:
    leaking = {
        "train": [make_tracking_id("seed_A", 0)],
        "test": [make_tracking_id("seed_A", 1)],  # same seed in both splits
    }
    assert not verify_no_leakage(leaking), "Failed to detect a deliberate leak"


def _run_all_tests_plainly() -> None:
    """Fallback runner if pytest isn't available in this environment."""
    test_functions = [
        obj for name, obj in list(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    passed, failed = 0, 0
    for fn in test_functions:
        try:
            fn()
            print(f"  [PASS] {fn.__name__}")
            passed += 1
        except AssertionError as exc:
            print(f"  [FAIL] {fn.__name__}: {exc}")
            failed += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  [ERROR] {fn.__name__}: {type(exc).__name__}: {exc}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {passed + failed}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    _run_all_tests_plainly()
