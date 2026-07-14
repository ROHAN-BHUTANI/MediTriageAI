"""Base classes and offline-safe helpers for the MediTriage model zoo."""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerBase

from src.model import MediTriageTransformer


def _cfg_value(config: Any, name: str, default: int) -> int:
    if config is None:
        return default
    if isinstance(config, dict):
        value = config.get(name, default)
    else:
        value = getattr(config, name, default)
    return int(value)


class SimpleClinicalTokenizer(PreTrainedTokenizer):
    """Minimal whitespace tokenizer used when pretrained tokenizers are unavailable."""

    def __init__(self, vocab: list[str] | None = None, **kwargs: Any) -> None:
        self._base_vocab = vocab or ["<pad>", " Ċ", "<cls>", "<sep>", "patient", "pain", "normal", "severe"]
        self._token_to_id = {token: index for index, token in enumerate(self._base_vocab)}
        self._id_to_token = {index: token for token, index in self._token_to_id.items()}
        super().__init__(pad_token="<pad>", unk_token=" Ċ", cls_token="<cls>", sep_token="<sep>", **kwargs)

    @property
    def vocab_size(self) -> int:
        return len(self._token_to_id)

    def get_vocab(self) -> dict[str, int]:
        vocab = dict(self._token_to_id)
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text: str) -> list[str]:
        tokens = re.findall(r"[A-Za-z0-9']+|[^\w\s]", text.lower())
        return tokens or [self.unk_token]

    def _convert_token_to_id(self, token: str) -> int:
        vocab = self.get_vocab()
        return vocab.get(token, vocab[self.unk_token])

    def _convert_id_to_token(self, index: int) -> str:
        vocab = self.get_vocab()
        for token, token_id in vocab.items():
            if token_id == index:
                return token
        return self.unk_token

    def build_inputs_with_special_tokens(self, token_ids_0: list[int], token_ids_1: list[int] | None = None) -> list[int]:
        cls_id = self.convert_tokens_to_ids(self.cls_token)
        sep_id = self.convert_tokens_to_ids(self.sep_token)
        if token_ids_1 is None:
            return [cls_id, *token_ids_0, sep_id]
        return [cls_id, *token_ids_0, sep_id, *token_ids_1, sep_id]

    def save_vocabulary(self, save_directory: str, filename_prefix: str | None = None):
        return ()


def load_tokenizer_or_fallback(model_name: str) -> PreTrainedTokenizerBase:
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(model_name, local_files_only=False)
    except Exception:
        return SimpleClinicalTokenizer()


def build_transformer_config(
    config: Any,
    tokenizer: PreTrainedTokenizerBase,
    *,
    hidden_size: int,
    num_hidden_layers: int,
    num_attention_heads: int,
    intermediate_size: int,
    max_position_embeddings: int,
    model_type: str,
) -> dict[str, int]:
    return {
        "vocab_size": len(tokenizer),
        "hidden_size": _cfg_value(config, "hidden_size", hidden_size),
        "num_hidden_layers": _cfg_value(config, "num_hidden_layers", num_hidden_layers),
        "num_attention_heads": _cfg_value(config, "num_attention_heads", num_attention_heads),
        "intermediate_size": _cfg_value(config, "intermediate_size", intermediate_size),
        "max_position_embeddings": _cfg_value(config, "max_position_embeddings", max_position_embeddings),
        "model_type": model_type,
    }


class BaseMediTriageModel(ABC):
    """Abstract base class for the four-model MediTriage zoo."""

    model_name: ClassVar[str]
    display_name: ClassVar[str]
    short_name: ClassVar[str]
    is_novel_contribution: ClassVar[bool] = False

    @classmethod
    @abstractmethod
    def build_encoder(cls, config: Any | None = None) -> PreTrainedModel:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def build_tokenizer(cls) -> PreTrainedTokenizerBase:
        raise NotImplementedError

    @classmethod
    def needs_vocab_injection(cls) -> bool:
        return True

    @classmethod
    def get_special_loading_notes(cls) -> str:
        return ""

    def get_tokenizer(self) -> PreTrainedTokenizerBase:
        """Get tokenizer for this model instance."""
        return self.build_tokenizer()

    def build(self, config: Any | None = None) -> MediTriageTransformer:
        """Build and return the dual-head model for this instance."""
        encoder = self.build_encoder(config)
        return MediTriageTransformer(encoder)

    @staticmethod
    def inject_vocab(model: MediTriageTransformer, tokenizer: PreTrainedTokenizerBase) -> int:
        """Inject Hinglish phonetic vocabulary into tokenizer and model embeddings."""
        from src.vocab_injection import build_vocab_injection_plan, inject_vocabulary_and_init_embeddings
        plan = build_vocab_injection_plan(tokenizer)
        return inject_vocabulary_and_init_embeddings(model, tokenizer, plan)


@dataclass(frozen=True)
class ZooConfig:
    hidden_size: int = 64
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    intermediate_size: int = 128
    max_position_embeddings: int = 512