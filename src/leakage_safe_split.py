"""Leakage-safe grouped split helpers for MediTriageAI."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field

SplitName = str


class InsufficientSeedsError(ValueError):
    pass


@dataclass(frozen=True)
class SplitAssignment:
    seed_id: str
    split: SplitName


@dataclass
class SplitResult:
    assignments: dict[str, SplitName]
    seed_counts: dict[SplitName, int] = field(default_factory=dict)


def make_tracking_id(seed_id: str, variant_index: int) -> str:
    basis = f"{seed_id}::{variant_index}".encode("utf-8")
    short_hash = hashlib.sha256(basis).hexdigest()[:8]
    return f"{seed_id}::v{variant_index}::{short_hash}"


def parse_seed_id_from_tracking_id(tracking_id: str) -> str:
    return tracking_id.split("::", 1)[0]


def compute_grouped_split(seed_ids: list[str], *, train_fraction: float = 0.8, val_fraction: float = 0.1, test_fraction: float = 0.1, random_seed: int = 1337, allow_empty_split: bool = False) -> SplitResult:
    if len(set(seed_ids)) != len(seed_ids):
        duplicates = [s for s in set(seed_ids) if seed_ids.count(s) > 1]
        raise ValueError(f"seed_ids must be unique; found {len(duplicates)} duplicate(s), e.g. {duplicates[:5]!r}.")
    fraction_sum = train_fraction + val_fraction + test_fraction
    if abs(fraction_sum - 1.0) > 1e-6:
        raise ValueError("train_fraction + val_fraction + test_fraction must sum to 1.0")
    n_seeds = len(seed_ids)
    if n_seeds == 0:
        raise InsufficientSeedsError("Cannot split an empty list of seed_ids.")
    for split_name, fraction in {"train": train_fraction, "val": val_fraction, "test": test_fraction}.items():
        if fraction <= 0.0:
            continue
        if round(n_seeds * fraction) == 0 and not allow_empty_split:
            raise InsufficientSeedsError(f"Requested {split_name} split rounds to 0 rows for {n_seeds} seeds.")
    rng = random.Random(random_seed)
    shuffled = list(seed_ids)
    rng.shuffle(shuffled)
    n_train = round(n_seeds * train_fraction)
    n_val = round(n_seeds * val_fraction)
    n_test = n_seeds - n_train - n_val
    if n_test < 0:
        n_train += n_test
        n_test = 0
    train_ids = shuffled[:n_train]
    val_ids = shuffled[n_train : n_train + n_val]
    test_ids = shuffled[n_train + n_val :]
    assignments = {sid: "train" for sid in train_ids} | {sid: "val" for sid in val_ids} | {sid: "test" for sid in test_ids}
    return SplitResult(assignments=assignments, seed_counts={"train": len(train_ids), "val": len(val_ids), "test": len(test_ids)})


def assign_rows_to_split(tracking_ids: list[str], split_result: SplitResult) -> dict[str, SplitName]:
    result: dict[str, SplitName] = {}
    for tracking_id in tracking_ids:
        seed_id = parse_seed_id_from_tracking_id(tracking_id)
        if seed_id not in split_result.assignments:
            raise KeyError(f"tracking_id {tracking_id!r} derives from seed_id {seed_id!r}, which was not included in the split computation.")
        result[tracking_id] = split_result.assignments[seed_id]
    return result


def verify_no_leakage(tracking_ids_by_split: dict[SplitName, list[str]]) -> bool:
    seed_to_splits: dict[str, set[SplitName]] = {}
    for split_name, tracking_ids in tracking_ids_by_split.items():
        for tracking_id in tracking_ids:
            seed_id = parse_seed_id_from_tracking_id(tracking_id)
            seed_to_splits.setdefault(seed_id, set()).add(split_name)
    return all(len(splits) == 1 for splits in seed_to_splits.values())
