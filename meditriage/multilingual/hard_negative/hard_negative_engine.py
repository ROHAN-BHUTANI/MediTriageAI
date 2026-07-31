"""Clinical Hard Negative Generation Engine.

Orchestrates differential diagnosis hard negative generation to train transformers
to distinguish highly similar emergency presentations.
"""

from __future__ import annotations

import logging
import random
from typing import Any

import pandas as pd

from meditriage.multilingual.hard_negative.hard_negative_config import (
    HardNegativeConfig,
)
from meditriage.multilingual.hard_negative.hard_negative_library import (
    DifferentialDiagnosisLibrary,
)
from meditriage.multilingual.hard_negative.hard_negative_report import (
    generate_hard_negative_reports,
)
from meditriage.multilingual.hard_negative.hard_negative_validator import (
    HardNegativeValidationResult,
    HardNegativeValidator,
)

logger = logging.getLogger(__name__)


class ClinicalHardNegativeEngine:
    """Core orchestrator for differential diagnosis hard negative generation."""

    CANONICAL_COLUMNS = [
        "id",
        "split",
        "dataset_source",
        "language",
        "raw_text",
        "department",
        "triage_level",
    ]

    def __init__(self, cfg: HardNegativeConfig | None = None):
        self.cfg = cfg or HardNegativeConfig()
        self.library = DifferentialDiagnosisLibrary()
        self.validator = HardNegativeValidator()
        self.stats: dict[str, Any] = {
            "total_source_records": 0,
            "total_negatives_generated": 0,
            "differential_counts": {},
            "validation_passed": 0,
            "validation_failed": 0,
        }

    def generate_hard_negatives_for_row(
        self, row: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Generate differential hard negative samples for a single input record.

        Args:
            row: Input dict representing one record.

        Returns:
            List of generated canonical row dicts.
        """
        orig_id = str(row.get("id"))
        orig_text = str(row.get("raw_text") or row.get("text") or "").strip()
        dept = str(row.get("department") or "GENERAL")
        split = row.get("split", "train")
        source = row.get("dataset_source", "unknown")
        lang = str(row.get("language") or "en")

        if not orig_text:
            return []

        differentials = self.library.get_differentials_for_text(orig_text, dept)
        if not differentials:
            return []

        rng = random.Random(self.cfg.random_seed + hash(orig_id) % 10000)
        generated_rows: list[dict[str, Any]] = []

        accepted_count = 0
        for diff_entry in differentials:
            if accepted_count >= self.cfg.negatives_per_sample:
                break

            candidate_pool = diff_entry.patient_wording + diff_entry.clinical_wording
            rng.shuffle(candidate_pool)

            for cand_text in candidate_pool:
                val_res: HardNegativeValidationResult = (
                    self.validator.validate_hard_negative(
                        source_text=orig_text,
                        negative_text=cand_text,
                        diff_entry=diff_entry,
                        original_department=dept,
                    )
                )

                if val_res.passed:
                    self.stats["validation_passed"] += 1
                    var_id = (
                        f"{orig_id}::hardneg_{diff_entry.diff_id}_{accepted_count + 1}"
                    )

                    new_row = {
                        "id": var_id,
                        "split": split,
                        "dataset_source": f"{source}_hardneg_{diff_entry.diff_id}",
                        "language": lang,
                        "raw_text": cand_text,
                        "department": diff_entry.target_department,
                        "triage_level": diff_entry.target_triage_level,
                    }
                    generated_rows.append(new_row)
                    accepted_count += 1

                    diff_name = diff_entry.name
                    self.stats["differential_counts"][diff_name] = (
                        self.stats["differential_counts"].get(diff_name, 0) + 1
                    )
                    break
                else:
                    self.stats["validation_failed"] += 1
                    logger.debug(
                        "Hard negative rejected (%s): %s",
                        diff_entry.name,
                        val_res.reason,
                    )

        return generated_rows

    def expand_dataframe(
        self, df: pd.DataFrame, preserve_original: bool = True
    ) -> pd.DataFrame:
        """Expand DataFrame with differential diagnosis hard negative samples.

        Args:
            df: Input DataFrame.
            preserve_original: If True, original input rows are included in output.

        Returns:
            DataFrame with hard negative rows added.
        """
        self.stats["total_source_records"] = len(df)
        logger.info(
            "Starting Clinical Hard Negative Engine on %d records (%d negatives/sample)",
            len(df),
            self.cfg.negatives_per_sample,
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

            negs = self.generate_hard_negatives_for_row(row)
            all_rows.extend(negs)

        out_df = pd.DataFrame(all_rows)

        # Enforce canonical schema order
        for col in self.CANONICAL_COLUMNS:
            if col not in out_df.columns:
                out_df[col] = None
        out_df = out_df[self.CANONICAL_COLUMNS].copy()

        self.stats["total_negatives_generated"] = len(out_df) - (
            len(df) if preserve_original else 0
        )

        # Generate reports
        generate_hard_negative_reports(out_df, self.stats, self.cfg)

        logger.info(
            "Hard Negative Generation complete: %d -> %d rows (%d negatives generated)",
            len(df),
            len(out_df),
            self.stats["total_negatives_generated"],
        )
        return out_df
