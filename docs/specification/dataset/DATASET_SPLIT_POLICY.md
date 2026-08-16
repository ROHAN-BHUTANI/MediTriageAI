# Dataset Split Policy

**Version:** v2.0.0
**Last Updated:** 2026-08-16
**Status:** ACTIVE — Governs all canonical dataset builds

---

## Algorithm

**Source-Aware Deterministic Stratified Split**

The canonical dataset uses a deterministic, source-aware split strategy that guarantees reproducibility and prevents data leakage.

### Procedure

1. **Group** all post-QC, post-dedup records by `source_dataset`.
2. **Within each source group**, sort records by SHA-256 hash of `source_record_id` (deterministic ordering independent of ingestion order).
3. **Assign** the first ~80% of sorted records to `train`, the next ~10% to `val`, and the remainder to `test`.
4. For sources with fewer than `min_source_for_stratification` (default: 10) records, assign **all** to `train` (documented tiny-source fallback).
5. Split assignment occurs **after** ingestion, QC, and deduplication, but **before** augmentation.

### Sort Key

```python
SHA-256(source_record_id)[:16 hex chars] / 0xFFFFFFFFFFFFFFFF → float ∈ [0, 1)
```

This provides a deterministic pseudo-random ordering that is:
- Independent of file order
- Independent of ingestion order
- Reproducible across runs
- Uniformly distributed

---

## Target Proportions

| Split | Target | Tolerance |
|-------|--------|-----------|
| train | 80%    | ±2%       |
| val   | 10%    | ±2%       |
| test  | 10%    | ±2%       |

These proportions are enforced **per source** (not globally), ensuring each source contributes proportionally to every split.

---

## Grouping Key

| Key | Purpose |
|-----|---------|
| `source_record_id` | Primary grouping key; all records with the same ID must be in the same split |
| `source_dataset` | Stratification key; each source is split independently |

---

## Stratification Keys

Stratification is applied at the **source level** (i.e., each source independently achieves ~80/10/10). Further stratification by department, triage_level, or language is **not applied** at the split level because:

1. Many sources have only one department (e.g., NHAMCS = ED only)
2. Most sources have no native triage_level labels
3. The pilot is 100% English; sub-stratification by language would be meaningless

If future builds include sufficiently diverse multilingual sources, language-level stratification may be added.

---

## Leakage Prevention

The following must **NEVER** cross splits:

| Entity | Rule |
|--------|------|
| Same `source_record_id` | Must all be in the same split |
| Same `source_record_id` parent | Augmented children inherit parent's split |
| Same augmentation family | Entire family (parent + all descendants) must share a split |
| Same patient/group identifier | Where available (e.g., CPSC_Case_Number) |
| Same exact normalized text | Enforced by deduplication before split assignment |

### Enforcement

1. **Deduplication** runs before split assignment, removing exact duplicates
2. **Split assignment** uses `source_record_id` as the grouping key
3. **Augmentation** (future stage) inherits the parent's split
4. **Leakage check** validates post-split that no `source_record_id` spans multiple splits

---

## Handling of Tiny Sources

Sources with fewer than 10 records are assigned entirely to `train`.

**Rationale:** With <10 records, splitting 80/10/10 would yield 0-1 records in val/test, which is statistically meaningless and could create empty strata.

This is a documented, deterministic behavior, not a bug.

---

## Handling of Unlabeled Severity

Records without native `triage_level` labels (the majority) are split by the same algorithm. No special handling is needed because:

1. Split is by `source_record_id`, not by label
2. The labeled subset (NHAMCS native_esi) is independently split ~80/10/10 within the NHAMCS source group
3. No severity-based stratification is applied at split time

---

## Handling of Multilingual Strata

Currently: all records are English. No multilingual stratification is performed.

When multilingual data is added:
- Each `source_dataset` will be independently split
- If a single source contains multiple languages, language distribution will be approximately preserved by the hash-based ordering (uniform distribution)
- Explicit language-level stratification may be added if needed

---

## Handling of Augmentation Parents

Augmentation is applied **after** split assignment.

Rules:
1. The augmented child record inherits the split of its parent (`augmentation_parent_id` → parent's split)
2. The `source_record_id` of augmented records should reference the parent's `source_record_id`
3. This ensures the entire augmentation family stays in one split

---

## Reproducibility Procedure

To reproduce the exact split:

1. Start from the same raw dataset files (checksummed)
2. Run the same `build_pilot.py` (or future `build_canonical.py`)
3. The SHA-256 sort key guarantees identical ordering
4. The round-based boundary calculation guarantees identical split boundaries
5. The output parquet is checksummed (SHA-256) for verification

### Deterministic Dependencies

| Component | Deterministic? |
|-----------|---------------|
| SHA-256 hash | Yes |
| Sort order | Yes (hash-based, no ties) |
| Boundary calculation | Yes (round arithmetic) |
| Seed/random state | Not used (fully deterministic) |
| File read order | Not relevant (sort key overrides) |

---

## Previous Strategy (DEPRECATED)

The initial pilot used a global SHA-256 threshold split:

```python
hash < 0.8 → train
hash < 0.9 → val
else → test
```

This was **deprecated** because:
1. When NEISS CPSC_Case_Number was NaN, all records got the same hash
2. Even with unique IDs, per-source proportions were not guaranteed
3. No source-level stratification

The current strategy replaces this with source-aware deterministic stratified assignment.
