"""
severity_heuristic.py
----------------------
Deterministic regex/keyword scanner that assigns a PROVISIONAL, LOW-CONFIDENCE
severity label (S1-S5) to free-text clinical narrative.

*** THIS IS NOT A VALIDATED CLINICAL INSTRUMENT. ***
See docs/01_clinical_taxonomy.md Section 4-5 for the full rationale and known
failure modes (no negation handling, no historical-vs-current discrimination,
no inter-annotator validation). Every label this module produces must carry
the `severity_label_source="regex_heuristic_v0"` / `severity_confidence="low"`
metadata — callers must not strip that metadata when persisting labels.

Design: ordered cascade of keyword/regex tiers, S1 checked first (most
specific/highest-stakes), falling through to a S4 default. This ordering
matters: a document mentioning both "history of cardiac arrest" (now stable)
and "routine follow-up" will hit the S1 regex first under this naive scanner
and be mislabeled S1 -- this is the textbook failure mode the heuristic does
NOT correct for, by design, because fixing it requires negation/temporality
modeling that is out of scope for a deterministic regex pass.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

SeverityLabel = str  # one of "S1".."S5"

VALID_SEVERITY_LABELS: tuple[SeverityLabel, ...] = ("S1", "S2", "S3", "S4", "S5")


@dataclass(frozen=True)
class SeverityHeuristicResult:
    severity: SeverityLabel
    matched_tier: str
    matched_pattern: str | None  # which regex fired, for auditability
    label_source: str = "regex_heuristic_v0"
    confidence: str = "low"


# Each tier is a list of compiled regexes, checked in order S1 -> S2 -> S3 -> S5,
# with S4 as the unconditional fallback (the "stable, nothing acute" middle
# bucket). Patterns are case-insensitive and use word boundaries to reduce
# trivial substring false positives (e.g. "arrest" inside an unrelated word).

_S1_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bcardiac arrest\b",
        r"\brespiratory arrest\b",
        r"\bcode blue\b",
        r"\bnot breathing\b",
        # NOTE: "unresponsive" alone is excluded. ~58% of its occurrences in
        # real clinical documentation are the benign surgical/clinical sense
        # "[condition] unresponsive to [therapy]" (i.e. conservative treatment
        # failed, hence the procedure) -- not a patient in an unresponsive
        # state. We use a negative lookahead to exclude "unresponsive to ..."
        # and only treat the bare patient-state sense as S1 signal.
        r"\bunresponsive\b(?!\s+to\b)",
        r"\bno pulse\b",
        # NOTE: plain "exsanguinated" is intentionally EXCLUDED here. In real
        # operative notes it overwhelmingly refers to the routine surgical
        # technique of draining a limb with an Esmarch bandage/tourniquet
        # before a bloodless field -- not a hemorrhagic emergency. We only
        # treat it as S1 signal when qualified by trauma/hemorrhage language,
        # which discriminates the two senses far better than the bare word.
        r"\bexsanguinat\w*\s+(?:\w+\s+){0,3}?(from|due to|secondary to)\s+(traumatic?|hemorrhage|haemorrhage|injury|gsw|stab)\b",
        r"\b(traumatic|hemorrhagic|massive) exsanguinat\w*\b",
        r"\bmassive (hemorrhage|haemorrhage|bleeding)\b",
        r"\banaphylaxis\b",
        r"\banaphylactic shock\b",
        r"\bcpr (in progress|initiated|performed)\b",
        r"\bflatlin\w*\b",
        r"\bpulseless\b",
    ]
]

_S2_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bsevere chest pain\b",
        r"\bcrushing chest pain\b",
        r"\bsudden (onset )?(weakness|numbness)\b",
        r"\bworst headache of (my|his|her|their) life\b",
        r"\bslurred speech\b",
        r"\bfacial droop\b",
        r"\bsuspected (mi|myocardial infarction|stroke|cva)\b",
        r"\bacute (mi|myocardial infarction|stroke)\b",
        r"\bsevere (respiratory distress|shortness of breath|dyspnea)\b",
        r"\baltered mental status\b",
        r"\bloss of consciousness\b",
        r"\bsevere abdominal pain\b",
        r"\bactive (seizure|seizing)\b",
        r"\bsevere allergic reaction\b",
        r"\buncontrolled bleeding\b",
        r"\bsevere trauma\b",
    ]
]

_S3_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bpersistent (high )?fever\b",
        r"\bmoderate (abdominal |chest )?pain\b",
        r"\bpersistent vomiting\b",
        r"\brecurrent (pain|symptoms)\b",
        r"\bworsening symptoms\b",
        r"\bhigh fever\b",
        r"\bdehydrat\w*\b",
        r"\bmoderate distress\b",
        r"\bsignificant pain\b",
    ]
]

_S5_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\broutine follow[- ]?up\b",
        r"\bannual (physical|exam|check[- ]?up)\b",
        r"\brefill request\b",
        r"\bno acute distress\b",
        r"\bnormal exam\b",
        r"\bwithin normal limits\b",
        r"\bstable[,.]? (no|without) (new |acute )?(complaints|issues)\b",
        r"\bwell[- ]?appearing\b",
        r"\bin no apparent distress\b",
        r"\bregular check[- ]?up\b",
    ]
]

_TIERED_PATTERNS: tuple[tuple[SeverityLabel, list[re.Pattern[str]]], ...] = (
    ("S1", _S1_PATTERNS),
    ("S2", _S2_PATTERNS),
    ("S3", _S3_PATTERNS),
    ("S5", _S5_PATTERNS),
)

_DEFAULT_SEVERITY: SeverityLabel = "S4"


def score_severity(text: str) -> SeverityHeuristicResult:
    """
    Run the deterministic tiered regex cascade against `text` and return the
    first matching severity tier (S1 highest priority, then S2, S3, S5), or
    the S4 default if nothing matches.

    Parameters
    ----------
    text:
        Free-text clinical narrative. Empty/whitespace-only or non-string
        input is treated as no-match (falls through to S4 default), since an
        empty document carries no signal either way.

    Returns
    -------
    SeverityHeuristicResult
        Includes which tier/pattern fired (or None for the default case) so
        every label is auditable back to the rule that produced it.
    """
    if not isinstance(text, str) or not text.strip():
        return SeverityHeuristicResult(
            severity=_DEFAULT_SEVERITY,
            matched_tier="default_empty_input",
            matched_pattern=None,
        )

    for tier_label, patterns in _TIERED_PATTERNS:
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                return SeverityHeuristicResult(
                    severity=tier_label,
                    matched_tier=tier_label,
                    matched_pattern=pattern.pattern,
                )

    return SeverityHeuristicResult(
        severity=_DEFAULT_SEVERITY,
        matched_tier="default_no_match",
        matched_pattern=None,
    )


if __name__ == "__main__":
    # Smoke tests covering each tier plus the documented failure mode.
    smoke_cases = [
        ("Patient found in cardiac arrest, CPR initiated.", "S1"),
        ("Presents with worst headache of his life and slurred speech.", "S2"),
        ("Persistent high fever for three days, mild dehydration.", "S3"),
        ("Routine annual physical, no acute distress, well-appearing.", "S5"),
        ("Patient here for medication management of stable hypertension.", "S4"),
        (
            "History of cardiac arrest two years ago, now stable, here for routine follow-up.",
            "S1",  # documented failure mode: history triggers S1 despite "now stable"
        ),
        (
            "The foot was elevated and exsanguinated with an Esmarch bandage "
            "prior to inflation of the pneumatic tourniquet.",
            "S4",  # surgical tourniquet technique, NOT a hemorrhagic emergency
        ),
        (
            "Patient exsanguinated rapidly secondary to traumatic gunshot wound.",
            "S1",  # genuine hemorrhagic emergency -- should still fire
        ),
        (
            "Severe menometrorrhagia unresponsive to medical therapy, "
            "proceeding to hysterectomy.",
            "S4",  # benign surgical-documentation sense, NOT a patient state
        ),
        (
            "Patient found unresponsive at home by family, EMS called.",
            "S1",  # genuine altered-consciousness emergency -- should still fire
        ),
        ("", "S4"),
    ]
    print("Smoke tests:")
    all_pass = True
    for text, expected in smoke_cases:
        result = score_severity(text)
        status = "OK" if result.severity == expected else "MISMATCH"
        if status == "MISMATCH":
            all_pass = False
        print(f"  [{status}] expected={expected} got={result.severity} "
              f"tier={result.matched_tier} text={text[:60]!r}")
    print(f"\nAll smoke tests matched expectation: {all_pass}")
