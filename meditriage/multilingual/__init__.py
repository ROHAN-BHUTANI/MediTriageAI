"""Multilingual Dataset Expansion Engine for MediTriageAI.

Provides natural multilingual clinical expansion across English, Hindi (Devanagari),
Roman Hindi, Natural Hinglish, and Code-Switched English-Hindi.
"""

from meditriage.multilingual.config import MultilingualConfig
from meditriage.multilingual.translator import MultilingualTranslator
from meditriage.multilingual.validator import ClinicalQualityValidator

__all__ = [
    "MultilingualConfig",
    "MultilingualTranslator",
    "ClinicalQualityValidator",
]
