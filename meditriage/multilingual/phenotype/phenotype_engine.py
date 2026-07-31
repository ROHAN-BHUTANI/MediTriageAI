"""Clinical Phenotype Augmentation Engine.

Orchestrates phenotype-level clinical augmentation across 8 medical specialties,
generating clinically valid symptom combinations representing identical disease phenotypes
while preserving ground-truth department and triage severity labels.
"""

from __future__ import annotations

import logging
import random
from typing import Any

import pandas as pd

from meditriage.multilingual.phenotype.phenotype_config import PhenotypeConfig
from meditriage.multilingual.phenotype.phenotype_library import PhenotypeDefinition, PhenotypeLibrary
from meditriage.multilingual.phenotype.phenotype_report import generate_phenotype_reports
from meditriage.multilingual.phenotype.phenotype_validator import (
    PhenotypeQualityValidator,
    PhenotypeValidationResult,
)

logger = logging.getLogger(__name__)


class ClinicalPhenotypeAugmentationEngine:
    """Core orchestrator for clinical phenotype augmentation."""

    CANONICAL_COLUMNS = [
        "id",
        "split",
        "dataset_source",
        "language",
        "raw_text",
        "department",
        "triage_level",
    ]

    def __init__(self, cfg: PhenotypeConfig | None = None):
        self.cfg = cfg or PhenotypeConfig()
        self.library = PhenotypeLibrary()
        self.validator = PhenotypeQualityValidator()
        self.stats: dict[str, Any] = {
            "total_source_records": 0,
            "total_variants_generated": 0,
            "phenotype_counts": {},
            "validation_passed": 0,
            "validation_failed": 0,
        }

    def generate_phenotype_variants_for_row(self, row: dict[str, Any]) -> list[dict[str, Any]]:
        """Generate phenotype variants for a single input record.

        Args:
            row: Input dict representing one record.

        Returns:
            List of generated canonical row dicts.
        """
        orig_id = str(row.get("id"))
        orig_text = str(row.get("raw_text") or row.get("text") or "").strip()
        dept = str(row.get("department") or "Emergency Medicine")
        triage = str(row.get("triage_level") or "S3")
        split = row.get("split", "train")
        source = row.get("dataset_source", "unknown")
        lang = str(row.get("language") or "en")

        if not orig_text:
            return []

        phenotype = self.library.match_phenotype(orig_text, dept)
        if not phenotype:
            return []

        # Check if phenotype's specialty is enabled
        if phenotype.specialty not in self.cfg.enabled_specialties:
            return []

        rng = random.Random(self.cfg.random_seed + hash(orig_id) % 10000)
        generated_rows: list[dict[str, Any]] = []

        # Candidate sources from phenotype definition
        candidate_pool = (
            phenotype.patient_wording
            + phenotype.clinical_wording
            + phenotype.triage_wording
        )

        # Synthesize core + supporting combinations
        for core in phenotype.core_symptoms:
            for opt in phenotype.optional_symptoms:
                candidate_pool.append(f"Patient reports {core} associated with {opt}.")
                candidate_pool.append(f"Complaint of {core} and {opt} since morning.")

        rng.shuffle(candidate_pool)

        accepted_count = 0
        for idx, candidate_text in enumerate(candidate_pool):
            if accepted_count >= self.cfg.variants_per_sample:
                break

            val_res: PhenotypeValidationResult = self.validator.validate_phenotype_variant(
                source_text=orig_text,
                variant_text=candidate_text,
                phenotype=phenotype,
                department=dept,
                triage_level=triage,
            )

            if val_res.passed:
                self.stats["validation_passed"] += 1
                var_id = f"{orig_id}::phenotype_{phenotype.phenotype_id}_{accepted_count+1}"

                new_row = {
                    "id": var_id,
                    "split": split,
                    "dataset_source": f"{source}_phenotype",
                    "language": lang,
                    "raw_text": candidate_text,
                    "department": dept,
                    "triage_level": triage,
                }
                generated_rows.append(new_row)
                accepted_count += 1

                spec = phenotype.specialty
                self.stats["phenotype_counts"][spec] = self.stats["phenotype_counts"].get(spec, 0) + 1
            else:
                self.stats["validation_failed"] += 1
                logger.debug("Phenotype variant rejected (%s): %s", phenotype.name, val_res.reason)

        return generated_rows

    def expand_dataframe(self, df: pd.DataFrame, preserve_original: bool = True) -> pd.DataFrame:
        """Expand DataFrame with clinical phenotype variants.

        Args:
            df: Input DataFrame.
            preserve_original: If True, original input rows are included in output.

        Returns:
            DataFrame with phenotype variants added.
        """
        self.stats["total_source_records"] = len(df)
        logger.info(
            "Starting Clinical Phenotype Augmentation Engine on %d records (%d variants/sample)",
            len(df), self.cfg.variants_per_sample
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

            variants = self.generate_phenotype_variants_for_row(row)
            all_rows.extend(variants)

        out_df = pd.DataFrame(all_rows)

        # Enforce canonical schema order
        for col in self.CANONICAL_COLUMNS:
            if col not in out_df.columns:
                out_df[col] = None
        out_df = out_df[self.CANONICAL_COLUMNS].copy()

        self.stats["total_variants_generated"] = len(out_df) - (len(df) if preserve_original else 0)

        # Generate reports
        generate_phenotype_reports(out_df, self.stats, self.cfg)

        logger.info(
            "Phenotype Augmentation complete: %d -> %d rows (%d variants generated)",
            len(df), len(out_df), self.stats["total_variants_generated"]
        )
        return out_df
