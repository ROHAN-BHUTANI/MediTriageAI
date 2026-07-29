"""XLM-RoBERTa-large entry in the MediTriage model zoo."""

from __future__ import annotations

from typing import Any

from transformers import AutoModel

from .base_model import BaseMediTriageModel, load_tokenizer_or_fallback


class XLMRobertaLargeModel(BaseMediTriageModel):
    model_name = "xlm-roberta-base"
    display_name = "XLM-RoBERTa-large"
    short_name = "xlm_roberta_large"
    is_novel_contribution = True

    @classmethod
    def build_tokenizer(cls):
        return load_tokenizer_or_fallback(cls.model_name)

    @classmethod
    def build_encoder(cls, config: Any | None = None):
        return AutoModel.from_pretrained(cls.model_name)
