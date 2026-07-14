"""mBERT entry in the MediTriage model zoo."""

from __future__ import annotations

from typing import Any

from transformers import BertConfig, BertModel

from .base_model import BaseMediTriageModel, ZooConfig, build_transformer_config, load_tokenizer_or_fallback


class MBertModel(BaseMediTriageModel):
    model_name = "bert-base-multilingual-cased"
    display_name = "mBERT"
    short_name = "mbert"

    @classmethod
    def build_tokenizer(cls):
        return load_tokenizer_or_fallback(cls.model_name)

    @classmethod
    def build_encoder(cls, config: Any | None = None):
        return BertModel.from_pretrained(cls.model_name)
