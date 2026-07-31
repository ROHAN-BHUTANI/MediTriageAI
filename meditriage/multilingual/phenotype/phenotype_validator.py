"""Phenotype Quality Validator for Clinical Phenotype Augmentation Engine.

Validates that generated phenotype variants preserve essential disease features,
pass rule engine physiological constraints, and maintain ground-truth department
and triage severity labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from meditriage.multilingual.phenotype.clinical_rules import ClinicalRuleEngine
from meditriage.multilingual.phenotype.phenotype_library import PhenotypeDefinition


@dataclass
class PhenotypeValidationResult:
    """Result of phenotype validation."""

    passed: bool
    reason: str = ""
    score: float = 1.0
    metrics: dict[str, Any] = field(default_factory=dict)


class PhenotypeQualityValidator:
    """Quality validator for clinical phenotype variants."""

    def __init__(self, rule_engine: ClinicalRuleEngine | None = None):
        self.rule_engine = rule_engine or ClinicalRuleEngine()

    def validate_phenotype_variant(
        self,
        source_text: str,
        variant_text: str,
        phenotype: PhenotypeDefinition,
        department: str | None = None,
        triage_level: str | None = None,
    ) -> PhenotypeValidationResult:
        """Validate a generated phenotype variant.

        Args:
            source_text: Original text.
            variant_text: Generated phenotype variant text.
            phenotype: Phenotype definition.
            department: Expected department.
            triage_level: Expected triage level.

        Returns:
            PhenotypeValidationResult instance.
        """
        # 1. Non-empty & length sanity check
        if not variant_text or not isinstance(variant_text, str):
            return PhenotypeValidationResult(passed=False, reason="Empty variant text", score=0.0)

        variant_clean = variant_text.strip()
        if len(variant_clean) < 10:
            return PhenotypeValidationResult(passed=False, reason="Variant text too short (< 10 chars)", score=0.0)

        # 2. Rule engine physiological verification
        passed_rules, rule_reason = self.rule_engine.validate_clinical_rules(
            variant_text=variant_clean,
            phenotype=phenotype,
            department=department,
            triage_level=triage_level,
        )
        if not passed_rules:
            return PhenotypeValidationResult(passed=False, reason=rule_reason, score=0.0)

        # 3. Phenotype core feature retention check
        # Variant must contain at least one symptom from core, optional, or supporting symptoms
        all_allowed_symptoms = (
            phenotype.core_symptoms
            + phenotype.optional_symptoms
            + phenotype.supporting_symptoms
        )
        var_lower = variant_clean.lower()
        matched = [s for s in all_allowed_symptoms if s.lower() in var_lower]

        # Also count if patient/clinical wording is used
        wording_matched = any(w.lower() in var_lower for w in phenotype.patient_wording)

        if not matched and not wording_matched:
            return PhenotypeValidationResult(
                passed=False,
                reason=f"Variant failed to retain any core symptom markers for phenotype '{phenotype.name}'",
                score=0.0,
            )

        quality_score = min(1.0, 0.5 + (len(matched) * 0.2))

        return PhenotypeValidationResult(
            passed=True,
            reason="Passed phenotype quality validation",
            score=round(quality_score, 2),
            metrics={
                "matched_symptoms": matched,
                "phenotype_id": phenotype.phenotype_id,
            },
        )
