"""DistilBERT multilingual entry in the MediTriage model zoo."""

from __future__ import annotations

from typing import Any

from transformers import DistilBertModel

from .base_model import BaseMediTriageModel, load_tokenizer_or_fallback


class DistilBertMultilingualModel(BaseMediTriageModel):
    model_name = "distilbert-base-multilingual-cased"
    display_name = "DistilBERT-multilingual"
    short_name = "distilbert_multilingual"

    @classmethod
    def build_tokenizer(cls):
        return load_tokenizer_or_fallback(cls.model_name)

    @classmethod
    def build_encoder(cls, config: Any | None = None):
        return DistilBertModel.from_pretrained(cls.model_name)
