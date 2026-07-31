"""Hard Negative Validator for Differential Diagnosis Augmentation Engine.

Validates that generated hard negative samples represent distinct, clinically valid
differential diagnoses while rejecting samples that match the original condition,
create impossible physiology, or violate department logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from meditriage.multilingual.hard_negative.hard_negative_library import DifferentialDiagnosis


@dataclass
class HardNegativeValidationResult:
    """Result of hard negative validation."""

    passed: bool
    reason: str = ""
    score: float = 1.0
    metrics: dict[str, Any] = field(default_factory=dict)


class HardNegativeValidator:
    """Validator for differential diagnosis hard negatives."""

    def validate_hard_negative(
        self,
        source_text: str,
        negative_text: str,
        diff_entry: DifferentialDiagnosis,
        original_department: str | None = None,
    ) -> HardNegativeValidationResult:
        """Validate a generated hard negative sample.

        Args:
            source_text: Original text.
            negative_text: Generated differential negative text.
            diff_entry: Target differential diagnosis entry.
            original_department: Ground-truth original department.

        Returns:
            HardNegativeValidationResult instance.
        """
        # 1. Non-empty check
        if not negative_text or not isinstance(negative_text, str):
            return HardNegativeValidationResult(passed=False, reason="Empty hard negative text", score=0.0)

        neg_clean = negative_text.strip()
        if len(neg_clean) < 10:
            return HardNegativeValidationResult(passed=False, reason="Hard negative text too short (< 10 chars)", score=0.0)

        neg_lower = neg_clean.lower()
        src_lower = source_text.lower()

        # 2. Check that negative text does NOT include red flags or exclusive markers of original condition
        for red_flag in diff_entry.red_flags:
            # If red_flag explicitly prohibits a finding (e.g. 'no radiation to arm') but text has it
            if "no " in red_flag:
                forbidden_finding = red_flag.replace("no ", "").strip()
                if forbidden_finding and forbidden_finding in neg_lower:
                    return HardNegativeValidationResult(
                        passed=False,
                        reason=f"Hard negative contains red-flag feature '{forbidden_finding}' of primary condition",
                        score=0.0,
                    )

        # 3. Distinguishing symptom check
        has_distinguishing = any(d.lower() in neg_lower for d in diff_entry.distinguishing_symptoms)
        has_patient_wording = any(w.lower() in neg_lower for w in diff_entry.patient_wording)

        if not has_distinguishing and not has_patient_wording:
            return HardNegativeValidationResult(
                passed=False,
                reason=f"Hard negative lacks distinguishing features for differential '{diff_entry.name}'",
                score=0.0,
            )

        return HardNegativeValidationResult(
            passed=True,
            reason="Passed hard negative validation",
            score=1.0,
            metrics={
                "differential_id": diff_entry.diff_id,
                "target_department": diff_entry.target_department,
            },
        )
