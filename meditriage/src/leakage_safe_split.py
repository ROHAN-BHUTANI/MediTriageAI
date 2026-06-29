"""
leakage_safe_split.py
------------------------
Leakage-safe train/validation/test split with unique tracking IDs.

THE LEAKAGE PROBLEM THIS SOLVES
--------------------------------
Each MTSamples row ("seed document") can spawn multiple derived rows in the
final dataset: the original English transcription, plus N synthetically
generated / phonetically perturbed Hinglish patient-self-report variants
derived from it (see src/hinglish_perturbation.py). If we split at the
*derived-row* level with a naive random split, near-duplicate variants of the
SAME underlying clinical seed document can land in both train and
validation/test -- the model then partly "memorizes" the seed document's
content during training and gets an inflated, illegitimate validation score
on a perturbed copy of something it already saw. This is textbook data
leakage via near-duplicate rows, and it is exactly the failure mode a
peer reviewer would flag first.

THE FIX
-------
We split at the SEED-DOCUMENT level (grouped split), not the derived-row
level. Every derived row carries:
  - `tracking_id`: a unique ID for that specific derived row.
  - `seed_id`: the ID of the original MTSamples row it was derived from.
All derived rows sharing a `seed_id` are assigned to the same split as a
group -- a seed document and ALL of its perturbed variants go entirely into
train, OR entirely into validation, OR entirely into test. Never split
across groups.

THE SMALL-N EDGE CASE THIS MODULE GUARDS AGAINST
-------------------------------------------------
If the number of distinct seed documents is small enough, naively computing
`val_count = int(round(n_seeds * val_fraction))` can round DOWN TO ZERO,
silently producing an empty validation (or test) split -- which then breaks
any downstream code that assumes a non-empty split (e.g. early stopping on
validation loss, or computing validation metrics at all). We explicitly
guard against this: if the rounded count for any non-empty requested split
would be zero, we raise rather than silently emitting an empty split, UNLESS
the caller passes `allow_empty_split=True` to acknowledge it explicitly.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field

SplitName = str  # one of "train", "val", "test"


class InsufficientSeedsError(ValueError):
    """Raised when the seed count is too small to produce a non-empty split."""


@dataclass(frozen=True)
class SplitAssignment:
    seed_id: str
    split: SplitName


@dataclass
class SplitResult:
    assignments: dict[str, SplitName]  # seed_id -> split
    seed_counts: dict[SplitName, int] = field(default_factory=dict)


def make_tracking_id(seed_id: str, variant_index: int) -> str:
    """
    Deterministic, collision-resistant tracking ID for a derived row.

    Parameters
    ----------
    seed_id:
        The originating seed document's stable identifier (e.g. the
        MTSamples row index, as a string).
    variant_index:
        0 for the original/unperturbed row derived from this seed;
        1, 2, 3, ... for successive perturbed variants of the same seed.

    Returns
    -------
    A tracking ID string of the form "{seed_id}::v{variant_index}::{hash8}".
    The trailing short hash is derived from (seed_id, variant_index) only --
    NOT from wall-clock time or any non-deterministic source -- so the same
    (seed_id, variant_index) pair always yields the same tracking_id, which
    is required for reproducible dataset rebuilds.
    """
    basis = f"{seed_id}::{variant_index}".encode("utf-8")
    short_hash = hashlib.sha256(basis).hexdigest()[:8]
    return f"{seed_id}::v{variant_index}::{short_hash}"


def parse_seed_id_from_tracking_id(tracking_id: str) -> str:
    """Inverse helper: extract the seed_id component from a tracking_id."""
    return tracking_id.split("::", 1)[0]


def compute_grouped_split(
    seed_ids: list[str],
    *,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    random_seed: int = 1337,
    allow_empty_split: bool = False,
) -> SplitResult:
    """
    Assign each distinct seed_id in `seed_ids` to exactly one of
    {"train", "val", "test"}, such that the three fractions are respected at
    the SEED level (every derived row sharing a seed_id then inherits that
    seed's split assignment -- enforced by the caller via `seed_id`, see
    `assign_rows_to_split` below).

    Guards against the documented small-N edge case: if the number of
    distinct seeds is small enough that a non-zero requested fraction would
    round down to 0 rows for val and/or test, this raises
    InsufficientSeedsError instead of silently producing an empty split.
    Pass allow_empty_split=True to explicitly opt out of this guard (e.g. for
    a deliberately train-only smoke-test run).

    Parameters
    ----------
    seed_ids:
        List of distinct seed document identifiers. Duplicates are an error
        (a seed_id must be unique -- if it isn't, the caller's seed-ID
        generation has a bug that needs fixing upstream, not silently
        deduplicating here).
    train_fraction, val_fraction, test_fraction:
        Must sum to 1.0 (within floating point tolerance).
    random_seed:
        Seeds a LOCAL random.Random instance (never global random state) for
        reproducible shuffling before the split boundaries are cut.
    allow_empty_split:
        If True, skips the zero-rows guard. Use only for intentional
        train-only or train+val-only smoke runs, and document why at the
        call site.

    Returns
    -------
    SplitResult with assignments keyed by seed_id, plus a summary of how many
    seeds landed in each split.
    """
    if len(set(seed_ids)) != len(seed_ids):
        duplicates = [s for s in set(seed_ids) if seed_ids.count(s) > 1]
        raise ValueError(
            f"seed_ids must be unique; found {len(duplicates)} duplicate(s), "
            f"e.g. {duplicates[:5]!r}. Fix seed-ID generation upstream."
        )

    fraction_sum = train_fraction + val_fraction + test_fraction
    if abs(fraction_sum - 1.0) > 1e-6:
        raise ValueError(
            f"train_fraction + val_fraction + test_fraction must sum to 1.0, "
            f"got {fraction_sum} (train={train_fraction}, val={val_fraction}, "
            f"test={test_fraction})"
        )

    n_seeds = len(seed_ids)
    if n_seeds == 0:
        raise InsufficientSeedsError("Cannot split an empty list of seed_ids.")

    # --- The small-N guard ---
    # For each split with a strictly positive requested fraction, check that
    # rounding does not produce zero rows. We check this BEFORE doing any
    # shuffling/assignment so the failure is loud and immediate.
    requested_fractions = {
        "train": train_fraction,
        "val": val_fraction,
        "test": test_fraction,
    }
    for split_name, fraction in requested_fractions.items():
        if fraction <= 0.0:
            continue  # split not requested at all; zero is fine/expected
        projected_count = round(n_seeds * fraction)
        if projected_count == 0 and not allow_empty_split:
            raise InsufficientSeedsError(
                f"Only {n_seeds} distinct seed document(s) available, but "
                f"the requested {split_name} fraction ({fraction}) would "
                f"round down to 0 rows for that split. This dataset is too "
                f"small for this split configuration -- either provide more "
                f"seed documents, reduce the number of splits requested "
                f"(e.g. train-only), or pass allow_empty_split=True if an "
                f"empty {split_name} split is genuinely intentional for this "
                f"run."
            )

    rng = random.Random(random_seed)  # local instance, not global random state
    shuffled = list(seed_ids)
    rng.shuffle(shuffled)

    n_train = round(n_seeds * train_fraction)
    n_val = round(n_seeds * val_fraction)
    # test gets the remainder, so rounding error doesn't drop/duplicate seeds
    n_test = n_seeds - n_train - n_val
    if n_test < 0:
        # Extremely small-N rounding edge case (e.g. n_seeds=1): clamp and
        # take from train rather than going negative.
        n_train += n_test
        n_test = 0

    train_ids = shuffled[:n_train]
    val_ids = shuffled[n_train : n_train + n_val]
    test_ids = shuffled[n_train + n_val :]

    assignments: dict[str, SplitName] = {}
    for sid in train_ids:
        assignments[sid] = "train"
    for sid in val_ids:
        assignments[sid] = "val"
    for sid in test_ids:
        assignments[sid] = "test"

    return SplitResult(
        assignments=assignments,
        seed_counts={"train": len(train_ids), "val": len(val_ids), "test": len(test_ids)},
    )


def assign_rows_to_split(
    tracking_ids: list[str],
    split_result: SplitResult,
) -> dict[str, SplitName]:
    """
    Given a list of derived-row tracking_ids and a SplitResult computed at
    the seed level, return the split assignment for each tracking_id by
    looking up its parent seed_id. This is what actually enforces
    "all variants of a seed go to the same split" at the row level.

    Raises KeyError if a tracking_id's parsed seed_id was not part of the
    seed_ids passed to compute_grouped_split (indicates a bug: every derived
    row's seed must have been included in the split computation).
    """
    result: dict[str, SplitName] = {}
    for tracking_id in tracking_ids:
        seed_id = parse_seed_id_from_tracking_id(tracking_id)
        if seed_id not in split_result.assignments:
            raise KeyError(
                f"tracking_id {tracking_id!r} derives from seed_id "
                f"{seed_id!r}, which was not included in the split "
                f"computation. Every seed must be split before its derived "
                f"rows are assigned."
            )
        result[tracking_id] = split_result.assignments[seed_id]
    return result


def verify_no_leakage(tracking_ids_by_split: dict[SplitName, list[str]]) -> bool:
    """
    Final safety check: verify that no seed_id appears in more than one
    split. Returns True iff the split is leakage-free. Intended to be called
    as a hard assertion right before persisting the final dataset splits to
    disk -- this is the test that would catch a regression if someone later
    "simplifies" the split logic back to a naive random row-level split.
    """
    seed_to_splits: dict[str, set[SplitName]] = {}
    for split_name, tracking_ids in tracking_ids_by_split.items():
        for tracking_id in tracking_ids:
            seed_id = parse_seed_id_from_tracking_id(tracking_id)
            seed_to_splits.setdefault(seed_id, set()).add(split_name)

    leaked = {sid: splits for sid, splits in seed_to_splits.items() if len(splits) > 1}
    return len(leaked) == 0


if __name__ == "__main__":
    print("=== Demo 1: normal-sized dataset ===")
    seed_ids = [f"mtsamples_{i}" for i in range(100)]
    result = compute_grouped_split(seed_ids, random_seed=42)
    print(f"Seed counts: {result.seed_counts}")

    # Build fake tracking IDs: 3 variants per seed (1 original + 2 perturbed)
    tracking_ids: list[str] = []
    for sid in seed_ids:
        for variant_idx in range(3):
            tracking_ids.append(make_tracking_id(sid, variant_idx))

    row_assignments = assign_rows_to_split(tracking_ids, result)
    by_split: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for tid, split in row_assignments.items():
        by_split[split].append(tid)
    print(f"Row counts (incl. 3x variants/seed): "
          f"{ {k: len(v) for k, v in by_split.items()} }")
    print(f"Leakage-free: {verify_no_leakage(by_split)}")

    print("\n=== Demo 2: small-N guard fires correctly ===")
    tiny_seed_ids = [f"seed_{i}" for i in range(5)]
    try:
        compute_grouped_split(tiny_seed_ids, train_fraction=0.8, val_fraction=0.1, test_fraction=0.1)
        print("ERROR: expected InsufficientSeedsError, none was raised!")
    except InsufficientSeedsError as exc:
        print(f"Correctly raised InsufficientSeedsError: {exc}")

    print("\n=== Demo 3: small-N guard allows explicit opt-out ===")
    result_small = compute_grouped_split(
        tiny_seed_ids, train_fraction=0.8, val_fraction=0.1, test_fraction=0.1,
        allow_empty_split=True,
    )
    print(f"With allow_empty_split=True: {result_small.seed_counts}")

    print("\n=== Demo 4: duplicate seed_ids rejected ===")
    try:
        compute_grouped_split(["a", "b", "a"])
        print("ERROR: expected ValueError, none was raised!")
    except ValueError as exc:
        print(f"Correctly raised ValueError: {exc}")
