from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from transformers import PreTrainedModel

from models.distilbert_multi import DistilBertMultilingualModel
from models.indic_bert import IndicBertModel
from models.mbert import MBertModel
from models.xlm_roberta import XLMRobertaLargeModel


@dataclass
class TinyConfig:
    hidden_size: int = 32
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    intermediate_size: int = 64
    max_position_embeddings: int = 64


@pytest.mark.parametrize(
    "model_cls,is_novel,display_name",
    [
        (XLMRobertaLargeModel, True, "XLM-RoBERTa-large"),
        (MBertModel, False, "mBERT"),
        (DistilBertMultilingualModel, False, "DistilBERT-multilingual"),
        (IndicBertModel, False, "IndicBERT"),
    ],
)
def test_model_metadata(model_cls, is_novel, display_name):
    model = model_cls()
    assert model.display_name == display_name
    assert model.is_novel_contribution is is_novel
    assert isinstance(model.model_name, str) and model.model_name
    assert isinstance(model.short_name, str) and model.short_name


def test_model_short_names_are_unique():
    short_names = {
        XLMRobertaLargeModel.short_name,
        MBertModel.short_name,
        DistilBertMultilingualModel.short_name,
        IndicBertModel.short_name,
    }
    assert len(short_names) == 4


@pytest.mark.parametrize(
    "model_cls",
    [XLMRobertaLargeModel, MBertModel, DistilBertMultilingualModel, IndicBertModel],
)
def test_model_classmethods_build_encoder(model_cls):
    encoder = model_cls.build_encoder()
    tokenizer = model_cls.build_tokenizer()
    assert isinstance(encoder, PreTrainedModel)
    assert tokenizer is not None
    assert isinstance(model_cls.needs_vocab_injection(), bool)
    assert isinstance(model_cls.get_special_loading_notes(), str)


@pytest.mark.parametrize(
    "model_cls",
    [XLMRobertaLargeModel, MBertModel, DistilBertMultilingualModel, IndicBertModel],
)
def test_model_tokenizer_and_forward_pass(model_cls):
    model = model_cls()
    tokenizer = model.get_tokenizer()
    built = model.build(TinyConfig())
    tokens = tokenizer(
        "patient has severe pain", return_tensors="pt", padding=True, truncation=True
    )
    specialist_logits, severity_logits = built(
        tokens["input_ids"], tokens["attention_mask"]
    )
    assert specialist_logits.shape[-1] == 13
    assert severity_logits.shape[-1] == 5
    assert specialist_logits.shape[0] == 1
    assert severity_logits.shape[0] == 1


def test_model_inject_vocab_returns_int():
    model = XLMRobertaLargeModel()
    tokenizer = model.get_tokenizer()
    built = model.build(TinyConfig())
    added = model.inject_vocab(built, tokenizer)
    assert isinstance(added, int)
    assert added >= 0


def test_forward_pass_is_deterministic_shape_only():
    model = MBertModel()
    tokenizer = model.get_tokenizer()
    built = model.build(TinyConfig())
    input_ids = torch.randint(0, len(tokenizer), (2, 8))
    attention_mask = torch.ones_like(input_ids)
    specialist_logits, severity_logits = built(input_ids, attention_mask)
    assert specialist_logits.shape == (2, 13)
    assert severity_logits.shape == (2, 5)
