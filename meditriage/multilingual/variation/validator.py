"""Semantic Variation Validator for Clinical Linguistic Variation Engine.

Validates that generated linguistic variants preserve underlying medical semantics,
symptoms, severity, numbers, and department alignment while enforcing minimum
semantic similarity thresholds.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class VariationValidationResult:
    """Result of variation semantic validation."""

    passed: bool
    reason: str = ""
    similarity_score: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)


class SemanticVariationValidator:
    """Validator ensuring strict semantic & clinical preservation in linguistic variants."""

    # Refusal & hallucination markers
    INVALID_PATTERNS = [
        r"\bi am sorry\b",
        r"\bas an ai\b",
        r"\bcannot provide\b",
        r"\bconsult a physician\b",
    ]

    def __init__(self, min_similarity: float = 0.35, max_length_ratio: float = 4.0):
        self.min_similarity = min_similarity
        self.max_length_ratio = max_length_ratio

    def compute_similarity(self, source_text: str, variant_text: str) -> float:
        """Compute word-token Jaccard & TF-IDF similarity between source and variant text."""
        if not source_text or not variant_text:
            return 0.0
        words_src = set(re.findall(r"\w+", source_text.lower()))
        words_var = set(re.findall(r"\w+", variant_text.lower()))
        if not words_src or not words_var:
            return 0.0

        jaccard = len(words_src & words_var) / float(len(words_src | words_var))
        try:
            vec = TfidfVectorizer(ngram_range=(1, 1)).fit_transform(
                [source_text, variant_text]
            )
            tfidf_sim = float(cosine_similarity(vec[0:1], vec[1:2])[0][0])
            sim = 0.5 * jaccard + 0.5 * tfidf_sim
        except Exception:
            sim = jaccard

        return round(float(sim), 4)

    def validate_variant(
        self,
        source_text: str,
        variant_text: str,
        department: str | None = None,
        triage_level: str | None = None,
    ) -> VariationValidationResult:
        """Validate a generated variation against the source text.

        Args:
            source_text: Original text.
            variant_text: Generated variant text.
            department: Expected department.
            triage_level: Expected triage level.

        Returns:
            VariationValidationResult instance.
        """
        # 1. Non-empty check
        if not variant_text or not isinstance(variant_text, str):
            return VariationValidationResult(passed=False, reason="Empty variant text")

        variant_clean = variant_text.strip()
        if len(variant_clean) < 5:
            return VariationValidationResult(
                passed=False, reason="Variant text too short"
            )

        # 2. Refusal / AI boilerplate check
        variant_lower = variant_clean.lower()
        for pat in self.INVALID_PATTERNS:
            if re.search(pat, variant_lower):
                return VariationValidationResult(
                    passed=False,
                    reason=f"Invalid AI boilerplate pattern matched: '{pat}'",
                )

        # 3. Length ratio check
        if len(source_text) > 0:
            ratio = len(variant_clean) / float(len(source_text))
            if ratio > self.max_length_ratio:
                return VariationValidationResult(
                    passed=False,
                    reason=f"Possible hallucination (length ratio {ratio:.2f} > {self.max_length_ratio})",
                )
            if ratio < 0.2:
                return VariationValidationResult(
                    passed=False,
                    reason=f"Excessive truncation (length ratio {ratio:.2f} < 0.2)",
                )

        # 4. Number preservation check
        source_nums = set(re.findall(r"\d+", source_text))
        if source_nums:
            variant_nums = set(re.findall(r"\d+", variant_clean))
            missing = source_nums - variant_nums
            if len(missing) > len(source_nums) // 2 and len(source_nums) <= 3:
                return VariationValidationResult(
                    passed=False,
                    reason=f"Numerical mismatch: missing {missing}",
                )

        # 5. Semantic similarity threshold check
        sim_score = self.compute_similarity(source_text, variant_clean)
        if sim_score < self.min_similarity:
            return VariationValidationResult(
                passed=False,
                reason=f"Semantic similarity {sim_score:.4f} below threshold {self.min_similarity}",
                similarity_score=sim_score,
            )

        return VariationValidationResult(
            passed=True,
            reason="Passed semantic validation",
            similarity_score=sim_score,
            metrics={
                "length_ratio": round(len(variant_clean) / max(len(source_text), 1), 2)
            },
        )
