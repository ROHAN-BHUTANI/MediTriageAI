"""IndicBERT entry in the MediTriage model zoo."""

from __future__ import annotations

from typing import Any

from transformers import BertConfig, BertModel

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
        tokenizer = cls.build_tokenizer()
        cfg = build_transformer_config(
            config,
            tokenizer,
            hidden_size=ZooConfig.hidden_size,
            num_hidden_layers=ZooConfig.num_hidden_layers,
            num_attention_heads=ZooConfig.num_attention_heads,
            intermediate_size=ZooConfig.intermediate_size,
            max_position_embeddings=ZooConfig.max_position_embeddings,
            model_type="bert",
        )
        return BertModel(BertConfig(**cfg, pad_token_id=0, type_vocab_size=2))
