import logging
import pandas as pd

from meditriage.multilingual.config import MultilingualConfig
from meditriage.multilingual.translator import MultilingualTranslator

logger = logging.getLogger(__name__)


def apply_augmentation(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Apply dataset augmentation / multilingual expansion.

    Args:
        df: Input DataFrame from Stage 4 (deduplicated).
        config: Configuration dict.

    Returns:
        Augmented/Expanded DataFrame.
    """
    if len(df) == 0:
        return df

    multilingual_config = config.get("multilingual", config.get("augmentation", {}))
    enabled = multilingual_config.get("enabled", True)
    target_langs = multilingual_config.get("target_languages", ["en", "hi", "hi-Latn", "hi-en", "en-hi"])

    if not enabled:
        return df

    provider_name = multilingual_config.get("provider", "offline")
    model_name = multilingual_config.get("model_name", "gemini-2.0-flash")

    cfg = MultilingualConfig(
        target_languages=target_langs,
        provider=provider_name,
        model_name=model_name,
        preserve_original=True,
    )

    translator = MultilingualTranslator(cfg)
    expanded_df = translator.expand_dataframe(df)

    logger.info("Stage 5 Augmentation: Expanded %d -> %d rows", len(df), len(expanded_df))
    return expanded_df

