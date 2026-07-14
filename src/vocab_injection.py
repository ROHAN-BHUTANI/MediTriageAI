"""Vocabulary injection for Hinglish phonetic variants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


@dataclass
class VocabInjectionPlan:
    vocab_size_before: int
    chunks_to_add: List[tuple[str, str]]
    init_mapping: dict[str, str]


def build_vocab_injection_plan(tokenizer) -> VocabInjectionPlan:
    from src.hinglish_perturbation import _VARIANT_TABLE

    vocab_size_before = len(tokenizer)
    chunks_to_add: list[tuple[str, str]] = []
    init_mapping: dict[str, str] = {}
    claimed_variants: set[str] = set()
    vocab = tokenizer.get_vocab()
    for variant_entry in _VARIANT_TABLE:
        anchor = variant_entry.canonical
        for variant in variant_entry.alternatives:
            if variant in vocab:
                continue
            chunks_to_add.append((anchor, variant))
            if variant not in claimed_variants:
                init_mapping[variant] = anchor
                claimed_variants.add(variant)
    return VocabInjectionPlan(vocab_size_before=vocab_size_before, chunks_to_add=chunks_to_add, init_mapping=init_mapping)


def inject_vocabulary_and_init_embeddings(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, plan: VocabInjectionPlan) -> int:
    vocab_size_before = len(tokenizer)
    tokenizer.add_tokens([variant for _, variant in plan.chunks_to_add])
    n_added = len(tokenizer) - vocab_size_before
    if hasattr(model, "resize_token_embeddings") and len(tokenizer) != vocab_size_before:
        model.resize_token_embeddings(len(tokenizer))
    if hasattr(model, "get_input_embeddings"):
        embedding_layer = model.get_input_embeddings()
        vocab = tokenizer.get_vocab()
        with torch.no_grad():
            for variant, anchor in plan.init_mapping.items():
                variant_id = vocab.get(variant)
                anchor_id = vocab.get(anchor)
                if variant_id is None or anchor_id is None or variant_id < plan.vocab_size_before or variant_id == anchor_id:
                    continue
                anchor_vector = embedding_layer.weight.data[anchor_id].clone()
                embedding_layer.weight.data[variant_id] = anchor_vector
    return n_added
