"""Clinical Consistency Rule Engine.

Enforces explicit physiological, organ-system, triage severity, and demographic
compatibility constraints on generated phenotype variants to prevent impossible
or clinically implausible symptom combinations.
"""

from __future__ import annotations

import re

from meditriage.multilingual.phenotype.phenotype_library import PhenotypeDefinition


class ClinicalRuleEngine:
    """Rule engine for verifying clinical consistency and physiological plausibility."""

    # Contradictory symptom pairs that can NEVER coexist in a valid single-presentation phenotype
    CONTRADICTORY_PAIRS = [
        (r"\bpregnancy\b", r"\bmale\b"),
        (r"\bpediatric croup\b", r"\bgeriatric\b"),
        (r"\bbilateral facial paralysis\b", r"\bappendicitis\b"),
        (r"\bfemur fracture\b", r"\bisolated migraine\b"),
        (r"\bacute myocardial infarction\b", r"\botitis externa\b"),
    ]

    # Department to organ system mappings
    DEPT_COMPATIBILITY = {
        "CARDIO_PULM": [
            "chest",
            "heart",
            "breath",
            "lung",
            "cardiac",
            "cough",
            "wheezing",
            "edema",
            "palpitation",
            "dyspnea",
            "pain",
            "substernal",
            "pressure",
            "discomfort",
        ],
        "NEURO": [
            "head",
            "brain",
            "headache",
            "seizure",
            "stroke",
            "weakness",
            "numbness",
            "dizziness",
            "droop",
            "aphasia",
        ],
        "ORTHO": [
            "bone",
            "joint",
            "fracture",
            "sprain",
            "dislocation",
            "swelling",
            "leg",
            "arm",
            "wrist",
            "ankle",
            "trauma",
            "pain",
        ],
        "PEDS": [
            "child",
            "pediatric",
            "fever",
            "cough",
            "vomiting",
            "feeding",
            "infant",
            "toddler",
            "crying",
            "lethargy",
        ],
        "ENT": [
            "ear",
            "nose",
            "throat",
            "hearing",
            "discharge",
            "sinus",
            "tonsil",
            "otalgia",
            "pain",
        ],
        "OBGYN": [
            "pregnancy",
            "labor",
            "vaginal",
            "pelvic",
            "maternal",
            "uterine",
            "fetal",
            "cramp",
        ],
    }

    def validate_clinical_rules(
        self,
        variant_text: str,
        phenotype: PhenotypeDefinition,
        department: str | None = None,
        triage_level: str | None = None,
    ) -> tuple[bool, str]:
        """Verify that variant_text satisfies all clinical rules.

        Args:
            variant_text: Generated variant string.
            phenotype: Target phenotype definition.
            department: Expected department.
            triage_level: Expected triage severity level.

        Returns:
            Tuple of (passed: bool, reason: str).
        """
        text_lower = variant_text.lower()

        # 1. Check explicit contraindicated symptoms for this phenotype
        for contra in phenotype.contraindicated_symptoms:
            if contra.lower() in text_lower:
                return (
                    False,
                    f"Contraindicated symptom '{contra}' present for phenotype '{phenotype.name}'",
                )

        # 2. Check contradictory pairs
        for pat1, pat2 in self.CONTRADICTORY_PAIRS:
            if re.search(pat1, text_lower) and re.search(pat2, text_lower):
                return (
                    False,
                    f"Contradictory clinical pair matched: '{pat1}' and '{pat2}'",
                )

        # 3. Check organ system / department compatibility
        dept_key = department or phenotype.department_mapping[0]
        if dept_key in self.DEPT_COMPATIBILITY:
            valid_keywords = self.DEPT_COMPATIBILITY[dept_key]
            # Must contain at least one organ system keyword compatible with department
            if not any(kw in text_lower for kw in valid_keywords):
                return (
                    False,
                    f"Text lacks organ system keywords compatible with department '{dept_key}'",
                )

        # 4. Check high-acuity triage compatibility
        if triage_level in ("S1", "S2"):
            # High acuity must not sound like a trivial complaint
            trivial_only = ["mild itch", "skin tag", "paper cut", "scratch"]
            if any(t in text_lower for t in trivial_only):
                return (
                    False,
                    f"Trivial symptom inconsistent with high-acuity triage level '{triage_level}'",
                )

        return True, "Passed clinical rule engine verification"
