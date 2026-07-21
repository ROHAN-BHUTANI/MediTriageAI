"""IndicBERT entry in the MediTriage model zoo."""

from __future__ import annotations

from typing import Any

from transformers import AutoModel, BertConfig, BertModel

from .base_model import BaseMediTriageModel, ZooConfig, build_transformer_config, load_tokenizer_or_fallback


class IndicBertModel(BaseMediTriageModel):
    model_name = "ai4bharat/indic-bert"
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
