"""Multilingual Translator Engine.

Orchestrates multilingual dataset expansion across languages, providers,
parallel workers, persistent caching, and clinical quality validation.
Output rows strictly conform to the 7 canonical columns:
  id, split, dataset_source, language, raw_text, department, triage_level
"""

from __future__ import annotations

import concurrent.futures
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd

from meditriage.multilingual.cache import MultilingualCache
from meditriage.multilingual.config import MultilingualConfig
from meditriage.multilingual.providers import get_provider
from meditriage.multilingual.validator import ClinicalQualityValidator, ValidationResult

from meditriage.multilingual.variation.config import VariationConfig
from meditriage.multilingual.variation.engine import ClinicalLinguisticVariationEngine

logger = logging.getLogger(__name__)


class MultilingualTranslator:
    """Core orchestrator for multilingual clinical dataset expansion."""

    CANONICAL_COLUMNS = [
        "id",
        "split",
        "dataset_source",
        "language",
        "raw_text",
        "department",
        "triage_level",
    ]

    def __init__(self, cfg: MultilingualConfig | None = None):
        self.cfg = cfg or MultilingualConfig()
        self.cache = MultilingualCache(cache_dir=self.cfg.cache_dir)
        self.validator = ClinicalQualityValidator()
        self.provider = get_provider(
            self.cfg.provider,
            model_name=self.cfg.model_name,
            max_retries=self.cfg.max_retries,
            initial_delay=self.cfg.initial_delay,
        )
        if self.cfg.enable_variations:
            var_cfg = (
                VariationConfig.from_dict(self.cfg.variation_config)
                if self.cfg.variation_config
                else VariationConfig()
            )
            self.variation_engine = ClinicalLinguisticVariationEngine(var_cfg)
        else:
            self.variation_engine = None

        if self.cfg.enable_phenotype_augmentation:
            from meditriage.multilingual.phenotype.phenotype_config import PhenotypeConfig
            from meditriage.multilingual.phenotype.phenotype_engine import ClinicalPhenotypeAugmentationEngine

            pheno_cfg = (
                PhenotypeConfig.from_dict(self.cfg.phenotype_config)
                if self.cfg.phenotype_config
                else PhenotypeConfig()
            )
            self.phenotype_engine = ClinicalPhenotypeAugmentationEngine(pheno_cfg)
        else:
            self.phenotype_engine = None

        if self.cfg.enable_hard_negatives:
            from meditriage.multilingual.hard_negative.hard_negative_config import HardNegativeConfig
            from meditriage.multilingual.hard_negative.hard_negative_engine import ClinicalHardNegativeEngine

            hn_cfg = (
                HardNegativeConfig.from_dict(self.cfg.hard_negative_config)
                if self.cfg.hard_negative_config
                else HardNegativeConfig()
            )
            self.hard_negative_engine = ClinicalHardNegativeEngine(hn_cfg)
        else:
            self.hard_negative_engine = None

        self.stats = {
            "total_input_rows": 0,
            "total_output_rows": 0,
            "language_counts": {},
            "validation_passed": 0,
            "validation_failed": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

    def _process_sample(self, row: dict[str, Any], target_lang: str) -> dict[str, Any] | None:
        """Process a single row for a specific target language."""
        orig_text = str(row.get("raw_text") or row.get("text") or "").strip()
        if not orig_text:
            return None

        # 1. Check cache first
        cached = self.cache.get(orig_text, target_lang, self.provider.name)
        if cached and cached.get("validated"):
            self.stats["cache_hits"] += 1
            translated_text = cached["translated_text"]
        else:
            self.stats["cache_misses"] += 1
            try:
                translated_text = self.provider.translate_text(
                    text=orig_text,
                    target_lang=target_lang,
                    department=row.get("department"),
                    triage_level=row.get("triage_level"),
                )
            except Exception as exc:
                logger.warning("Translation failed for ID %s (lang=%s): %s", row.get("id"), target_lang, exc)
                return None

            # 2. Quality validation
            val_res: ValidationResult = self.validator.validate(
                source_text=orig_text,
                target_text=translated_text,
                target_lang=target_lang,
                department=row.get("department"),
                triage_level=row.get("triage_level"),
            )

            if not val_res.passed:
                self.stats["validation_failed"] += 1
                logger.warning("Quality check failed for ID %s (lang=%s): %s", row.get("id"), target_lang, val_res.reason)
                if self.cfg.strict_validation:
                    return None
            else:
                self.stats["validation_passed"] += 1

            # Cache successful translation
            self.cache.set(
                text=orig_text,
                target_lang=target_lang,
                provider_name=self.provider.name,
                translated_text=translated_text,
                validated=val_res.passed,
                metrics=val_res.metrics,
            )

        # 3. Construct canonical output row
        new_id = f"{row.get('id', 'sample')}::{target_lang}"
        new_row = {
            "id": new_id,
            "split": row.get("split", "train"),
            "dataset_source": row.get("dataset_source", "unknown"),
            "language": target_lang,
            "raw_text": translated_text,
            "department": row.get("department"),
            "triage_level": row.get("triage_level"),
        }
        return new_row

    def expand_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Expand a DataFrame across all configured target languages.

        Args:
            df: Input DataFrame.

        Returns:
            Expanded DataFrame containing canonical 7 columns.
        """
        t0 = time.time()
        self.stats["total_input_rows"] = len(df)
        logger.info("Starting multilingual expansion on %d input rows", len(df))

        expanded_rows = []

        # Convert DF rows to dicts
        rows = df.to_dict("records")

        for row in rows:
            # 1. Optionally preserve original English row
            if self.cfg.preserve_original:
                orig_row = {
                    "id": str(row.get("id")),
                    "split": row.get("split", "train"),
                    "dataset_source": row.get("dataset_source", "unknown"),
                    "language": str(row.get("language") or "en"),
                    "raw_text": str(row.get("raw_text") or row.get("text") or ""),
                    "department": row.get("department"),
                    "triage_level": row.get("triage_level"),
                }
                expanded_rows.append(orig_row)

            # 2. Expand for non-English target languages
            non_en_langs = [l for l in self.cfg.target_languages if l != "en"]

            if self.cfg.num_workers > 1 and len(non_en_langs) > 1:
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.cfg.num_workers) as executor:
                    futures = [
                        executor.submit(self._process_sample, row, lang)
                        for lang in non_en_langs
                    ]
                    for future in concurrent.futures.as_completed(futures):
                        res = future.result()
                        if res:
                            expanded_rows.append(res)
            else:
                for lang in non_en_langs:
                    res = self._process_sample(row, lang)
                    if res:
                        expanded_rows.append(res)

        # Save cache
        self.cache.save()

        # Build final DataFrame
        out_df = pd.DataFrame(expanded_rows)

        # Ensure canonical column order
        for col in self.CANONICAL_COLUMNS:
            if col not in out_df.columns:
                out_df[col] = None
        out_df = out_df[self.CANONICAL_COLUMNS].copy()

        # If variation engine is enabled, apply clinical linguistic variation
        if self.variation_engine is not None:
            out_df = self.variation_engine.expand_dataframe(out_df, preserve_original=True)

        # If phenotype augmentation engine is enabled, apply clinical phenotype augmentation
        if self.phenotype_engine is not None:
            out_df = self.phenotype_engine.expand_dataframe(out_df, preserve_original=True)

        # If hard negative engine is enabled, apply differential diagnosis hard negative generation
        if self.hard_negative_engine is not None:
            out_df = self.hard_negative_engine.expand_dataframe(out_df, preserve_original=True)

        self.stats["total_output_rows"] = len(out_df)
        self.stats["elapsed_seconds"] = round(time.time() - t0, 2)
        if "language" in out_df.columns:
            self.stats["language_counts"] = out_df["language"].value_counts().to_dict()

        logger.info(
            "Multilingual & Variation expansion complete: %d -> %d rows in %.2fs",
            len(df), len(out_df), self.stats["elapsed_seconds"]
        )
        return out_df
