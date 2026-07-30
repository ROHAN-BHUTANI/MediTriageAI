"""Deterministic severity heuristic for MediTriageAI."""

from __future__ import annotations

import re
from dataclasses import dataclass

SeverityLabel = str
VALID_SEVERITY_LABELS: tuple[SeverityLabel, ...] = ("S1", "S2", "S3", "S4", "S5")


@dataclass(frozen=True)
class SeverityHeuristicResult:
    severity: SeverityLabel
    matched_tier: str
    matched_pattern: str | None
    label_source: str = "regex_heuristic_v0"
    confidence: str = "low"


_S1_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bcardiac arrest\b",
        r"\brespiratory arrest\b",
        r"\bcode blue\b",
        r"\bnot breathing\b",
        r"\bunresponsive\b(?!\s+to\b)",
        r"\bno pulse\b",
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
_S2_PATTERNS = [
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
_S3_PATTERNS = [
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
_S5_PATTERNS = [
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
_TIERED_PATTERNS = (
    ("S1", _S1_PATTERNS),
    ("S2", _S2_PATTERNS),
    ("S3", _S3_PATTERNS),
    ("S5", _S5_PATTERNS),
)
_DEFAULT_SEVERITY = "S4"


def score_severity(text: str) -> SeverityHeuristicResult:
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
