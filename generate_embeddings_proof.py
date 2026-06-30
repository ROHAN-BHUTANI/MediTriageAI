from __future__ import annotations

import torch
from transformers import AutoTokenizer, XLMRobertaModel


MODEL_NAME = "xlm-roberta-large"
CUSTOM_TOKENS = ["drd", "bkar", "shans", "khasi", "bht"]
ANCHOR_TOKEN = "dard"


def _resolve_single_token_id(tokenizer: AutoTokenizer, token_text: str) -> int:
    token_ids = tokenizer.encode(token_text, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(
            f"Anchor token {token_text!r} must resolve to exactly one token in {MODEL_NAME}, "
            f"but it resolved to ids={token_ids}."
        )
    return token_ids[0]


def generate_embedding_proof_matrix() -> None:
    print("=" * 88)
    print("MediTriageAI | XLM-RoBERTa-Large Embedding Proof Matrix")
    print("=" * 88)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = XLMRobertaModel.from_pretrained(MODEL_NAME)
    model.eval()

    base_vocab_size = len(tokenizer)
    anchor_id = _resolve_single_token_id(tokenizer, ANCHOR_TOKEN)
    added_count = tokenizer.add_tokens(CUSTOM_TOKENS)
    model.resize_token_embeddings(len(tokenizer))

    embedding_layer = model.get_input_embeddings()

    noisy_id = tokenizer.convert_tokens_to_ids("drd")
    if noisy_id is None or noisy_id == tokenizer.unk_token_id:
        raise ValueError("Token 'drd' was not added to the tokenizer vocabulary.")

    with torch.no_grad():
        embedding_layer.weight[noisy_id].copy_(embedding_layer.weight[anchor_id])

    print(f"Model name           : {MODEL_NAME}")
    print(f"Base vocab size      : {base_vocab_size}")
    print(f"Tokens requested     : {len(CUSTOM_TOKENS)}")
    print(f"Tokens actually added : {len(tokenizer) - base_vocab_size}")
    print(f"Final vocab size     : {len(tokenizer)}")
    print(f"Embedding dimensions : {embedding_layer.weight.shape[1]}")
    print("-" * 88)

    header = f"{'Token':<10} {'Token ID':<10} {'Shape':<14} {'First 5 Weight Values':<70}"
    print(header)
    print("-" * 88)

    for token in CUSTOM_TOKENS:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id == tokenizer.unk_token_id:
            raise ValueError(f"Token {token!r} was not assigned a valid vocabulary id.")

        weight_tensor = embedding_layer.weight[token_id].detach().cpu()
        first_five = [round(float(value), 6) for value in weight_tensor[:5].tolist()]
        shape_text = str(tuple(weight_tensor.shape))

        print(
            f"{token:<10} {token_id:<10} {shape_text:<14} {first_five!s:<70}"
        )

    print("-" * 88)
    print(
        f"Anchor copy check: {ANCHOR_TOKEN!r} -> 'drd' | "
        f"source_id={anchor_id}, target_id={noisy_id}, "
        f"vectors_equal={torch.allclose(embedding_layer.weight[noisy_id], embedding_layer.weight[anchor_id])}"
    )
    print("Proof matrix complete.")


if __name__ == "__main__":
    generate_embedding_proof_matrix()