# MediTriageAI — Current Specification Baseline

## Authoritative Specification

**Version:** v1.0.0

**Status:** FROZEN

**Authoritative location:** `docs/specification/frozen/v1.0.0/`

---

## Important Governance Notes

1. **Specification freeze does NOT imply implementation completion.** Implementation status for every requirement is tracked independently in `TRACEABILITY.md` using the status taxonomy: IMPLEMENTED, PARTIAL, MISSING, UNKNOWN, MIGRATION, RESEARCH-EXPERIMENTAL, TBD. The freeze fixes *what* each requirement means, not whether it is currently satisfied.

2. **Implementation status remains governed by the status taxonomy** defined in Section 0 of `SPECIFICATION.md`. Freezing the specification locks the requirements, architecture, governance model, acceptance criteria, and change-control contract — not the implementation state.

3. **Changes require the approved change-control process.** Any modification to a requirement, interface, architectural decision, naming policy, or acceptance criterion described in the frozen baseline requires an explicit Change Request per SPEC-15, human approval, and a new specification version. No autonomous agent (GSD, Ralph, CodeRabbit, or any DGX execution) may alter this baseline directly.

---

## Package Contents

| File | Description |
|---|---|
| `SPECIFICATION.md` | Complete consolidated specification (candidate + amendments) |
| `TRACEABILITY.md` | Full FR-ID traceability matrix (18 FR-IDs) |
| `RISK_REGISTER.md` | Consolidated risk register (13 risks) |
| `FREEZE_RECORD.md` | Freeze metadata, timestamps, commit references |
| `ADRs/ADR-001.md` – `ADRs/ADR-012.md` | Architectural Decision Records (12 ADRs) |
| `MANIFEST.json` | Cryptographic verification manifest |
| `SHA256SUMS` | Flat checksum list for external verification |
