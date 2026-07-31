"""Clinical Linguistic Variation Engine.

Orchestrates multi-style clinical linguistic variation generation across 10 styles,
enforcing semantic preservation, quality validation, and 7-canonical-column schema format.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from meditriage.multilingual.variation.config import VariationConfig
from meditriage.multilingual.variation.generators import (
    get_all_generators,
    get_generator_by_name,
)
from meditriage.multilingual.variation.report import generate_variation_reports
from meditriage.multilingual.variation.validator import (
    SemanticVariationValidator,
    VariationValidationResult,
)

logger = logging.getLogger(__name__)


class ClinicalLinguisticVariationEngine:
    """Engine for producing multi-style clinical linguistic variations."""

    CANONICAL_COLUMNS = [
        "id",
        "split",
        "dataset_source",
        "language",
        "raw_text",
        "department",
        "triage_level",
    ]

    def __init__(self, cfg: VariationConfig | None = None):
        self.cfg = cfg or VariationConfig()
        self.validator = SemanticVariationValidator(
            min_similarity=self.cfg.min_semantic_similarity
        )
        self.generators = {
            name: get_generator_by_name(name) for name in self.cfg.enabled_styles
        }
        self.stats: dict[str, Any] = {
            "total_source_records": 0,
            "total_variants_generated": 0,
            "style_counts": {},
            "validation_passed": 0,
            "validation_failed": 0,
            "similarity_scores": [],
        }

    def generate_variations_for_row(self, row: dict[str, Any]) -> list[dict[str, Any]]:
        """Generate all enabled linguistic variations for a single input record."""
        orig_id = str(row.get("id"))
        orig_text = str(row.get("raw_text") or row.get("text") or "").strip()
        dept = row.get("department")
        triage = row.get("triage_level")
        split = row.get("split", "train")
        source = row.get("dataset_source", "unknown")
        lang = str(row.get("language") or "en")

        if not orig_text:
            return []

        generated_rows: list[dict[str, Any]] = []

        total_sample_variants = 0
        for style_name, generator in self.generators.items():
            if total_sample_variants >= self.cfg.max_variants_per_sample:
                break

            budget = self.cfg.variation_budgets.get(style_name, 1)
            if budget <= 0:
                continue

            # Generate variants in this style
            seed = self.cfg.random_seed + hash(orig_id) % 10000 + len(generated_rows)
            candidates = generator.generate_variants(
                text=orig_text,
                department=dept,
                triage_level=triage,
                budget=budget,
                seed=seed,
            )

            for idx, candidate_text in enumerate(candidates):
                # Validate semantic preservation
                val_res: VariationValidationResult = self.validator.validate_variant(
                    source_text=orig_text,
                    variant_text=candidate_text,
                    department=dept,
                    triage_level=triage,
                )

                if val_res.passed:
                    self.stats["validation_passed"] += 1
                    self.stats["similarity_scores"].append(val_res.similarity_score)

                    var_id = f"{orig_id}::var_{style_name}_{idx+1}"
                    new_row = {
                        "id": var_id,
                        "split": split,
                        "dataset_source": f"{source}_var_{style_name}",
                        "language": lang,
                        "raw_text": candidate_text,
                        "department": dept,
                        "triage_level": triage,
                    }
                    generated_rows.append(new_row)

                    self.stats["style_counts"][style_name] = (
                        self.stats["style_counts"].get(style_name, 0) + 1
                    )
                    total_sample_variants += 1
                else:
                    self.stats["validation_failed"] += 1
                    logger.debug("Variant rejected (%s): %s", style_name, val_res.reason)

                if total_sample_variants >= self.cfg.max_variants_per_sample:
                    break

        return generated_rows

    def expand_dataframe(self, df: pd.DataFrame, preserve_original: bool = True) -> pd.DataFrame:
        """Expand DataFrame with clinical linguistic variations across all enabled styles.

        Args:
            df: Input DataFrame.
            preserve_original: If True, original input rows are included in the output.

        Returns:
            DataFrame with variation rows added.
        """
        self.stats["total_source_records"] = len(df)
        logger.info(
            "Starting Clinical Linguistic Variation Engine on %d records (%d styles enabled)",
            len(df), len(self.generators)
        )

        all_rows: list[dict[str, Any]] = []
        records = df.to_dict("records")

        for row in records:
            if preserve_original:
                orig_row = {
                    "id": str(row.get("id")),
                    "split": row.get("split", "train"),
                    "dataset_source": row.get("dataset_source", "unknown"),
                    "language": str(row.get("language") or "en"),
                    "raw_text": str(row.get("raw_text") or row.get("text") or ""),
                    "department": row.get("department"),
                    "triage_level": row.get("triage_level"),
                }
                all_rows.append(orig_row)

            variants = self.generate_variations_for_row(row)
            all_rows.extend(variants)

        out_df = pd.DataFrame(all_rows)

        # Enforce canonical schema order
        for col in self.CANONICAL_COLUMNS:
            if col not in out_df.columns:
                out_df[col] = None
        out_df = out_df[self.CANONICAL_COLUMNS].copy()

        self.stats["total_variants_generated"] = len(out_df) - (len(df) if preserve_original else 0)
        total_eval = self.stats["validation_passed"] + self.stats["validation_failed"]
        self.stats["validation_pass_rate"] = (
            (self.stats["validation_passed"] / total_eval * 100.0) if total_eval > 0 else 100.0
        )

        # Generate reports
        generate_variation_reports(out_df, self.stats, self.cfg)

        logger.info(
            "Clinical Variation Engine complete: %d -> %d rows (%d variants generated)",
            len(df), len(out_df), self.stats["total_variants_generated"]
        )
        return out_df
