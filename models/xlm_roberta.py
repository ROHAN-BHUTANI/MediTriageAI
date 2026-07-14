"""XLM-RoBERTa-large entry in the MediTriage model zoo."""

from __future__ import annotations

from typing import Any

from transformers import XLMRobertaConfig, XLMRobertaModel

from .base_model import BaseMediTriageModel, ZooConfig, build_transformer_config, load_tokenizer_or_fallback


class XLMRobertaLargeModel(BaseMediTriageModel):
    model_name = "xlm-roberta-large"
    display_name = "XLM-RoBERTa-large"
    short_name = "xlm_roberta_large"
    is_novel_contribution = True

    @classmethod
    def build_tokenizer(cls):
        return load_tokenizer_or_fallback(cls.model_name)

    @classmethod
    def build_encoder(cls, config: Any | None = None):
        tokenizer = cls.build_tokenizer()
        cfg = build_transformer_config(
            config,
            tokenizer,
            hidden_size=ZooConfig.hidden_size,
            num_hidden_layers=ZooConfig.num_hidden_layers,
            num_attention_heads=ZooConfig.num_attention_heads,
            intermediate_size=ZooConfig.intermediate_size,
            max_position_embeddings=ZooConfig.max_position_embeddings,
            model_type="xlm-roberta",
        )
        return XLMRobertaModel(XLMRobertaConfig(**cfg, bos_token_id=0, eos_token_id=2, pad_token_id=1))
