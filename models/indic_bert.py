"""IndicBERT entry in the MediTriage model zoo."""

from __future__ import annotations

from typing import Any

from transformers import AutoModel

from .base_model import BaseMediTriageModel, load_tokenizer_or_fallback


class IndicBertModel(BaseMediTriageModel):
    model_name = "google/muril-base-cased"
    display_name = "IndicBERT"
    short_name = "indic_bert"

    @classmethod
    def build_tokenizer(cls):
        return load_tokenizer_or_fallback(cls.model_name)

    @classmethod
    def needs_vocab_injection(cls) -> bool:
        return False

    @classmethod
    def get_special_loading_notes(cls) -> str:
        return "IndicBERT uses AlbertTokenizer; vocab injection may produce fewer matched anchors on Hindi subwords."

    @classmethod
    def build_encoder(cls, config: Any | None = None):
        return AutoModel.from_pretrained(cls.model_name)
