"""
test_train_dual_head_model.py
--------------------------------
Test suite for scripts/train_dual_head_model.py.

ENVIRONMENT CAVEAT: this container has no GPU and huggingface.co is blocked
by network egress policy, so we cannot download the real xlm-roberta-large
checkpoint or its tokenizer here. These tests instead build a tiny,
genuinely-functional XLM-R-architecture stand-in (small hidden size/layers,
a real locally-trained SentencePiece vocabulary covering our actual
Hinglish canonical words) to exercise every code path with real tensors and
real tokenization -- not mocks of the forward pass itself. When run in an
environment with internet access, swapping in xlm-roberta-large requires no
code changes (see build_model_and_tokenizer in the script under test).

Run with: python3 -m pytest tests/test_train_dual_head_model.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import train_dual_head_model as tdm  # noqa: E402

sentencepiece = pytest.importorskip("sentencepiece")
from transformers import XLMRobertaConfig, XLMRobertaModel, XLMRobertaTokenizer  # noqa: E402

TINY_SPM_MODEL_PATH = Path("/tmp/test_tiny_xlmr_tokenizer.model")


# ===========================================================================
# Fixtures: a small but genuinely real SentencePiece vocab + XLM-R stand-in
# ===========================================================================


@pytest.fixture(scope="module")
def _trained_spm_model_path() -> Path:
    """
    Trains the underlying SentencePiece model file ONCE per test module run
    (this part is expensive and stateless -- safe to share). The
    XLMRobertaTokenizer object built FROM this file is constructed fresh per
    test via the tiny_tokenizer fixture below, since add_tokens() mutates a
    tokenizer in place and a shared, already-injected tokenizer would leak
    state between tests that are each supposed to start from a clean vocab.
    """
    plan = tdm.build_vocab_injection_plan()
    canonical_words = list(plan.canonical_to_variants.keys())

    corpus_lines = [
        "the patient presents with severe chest pain and shortness of breath",
        "routine follow up examination no acute distress noted",
        "patient was taken to the operating room for surgery",
        "history of diabetes hypertension and coronary artery disease",
        "mera bahut dard ho raha hai kal subah se theek nahi hun",
        "aapka tabiyat kaisi hai hospital jana padega kya",
        "yeh dard bohot zyada hai raat ko sone nahi de raha",
    ] + canonical_words * 20

    corpus_path = Path("/tmp/test_spm_corpus.txt")
    corpus_path.write_text("\n".join(corpus_lines) + "\n")

    sentencepiece.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=str(TINY_SPM_MODEL_PATH.with_suffix("")),
        vocab_size=300,
        model_type="bpe",
        pad_id=1,
        unk_id=3,
        bos_id=0,
        eos_id=2,
        user_defined_symbols=["<mask>"],
    )
    return TINY_SPM_MODEL_PATH


@pytest.fixture
def tiny_tokenizer(_trained_spm_model_path: Path) -> XLMRobertaTokenizer:
    """
    Builds a FRESH XLMRobertaTokenizer instance per test from the (shared,
    already-trained) SentencePiece model file. Function-scoped deliberately:
    add_tokens() mutates a tokenizer's vocabulary in place, so reusing one
    tokenizer instance across tests that each call
    inject_vocabulary_and_init_embeddings would leak added tokens from one
    test into the next and desync it from a freshly-built, unresized
    tiny_encoder fixture -- which is exactly the bug this comment is here to
    prevent reintroducing.
    """
    sp = sentencepiece.SentencePieceProcessor(model_file=str(_trained_spm_model_path))
    vocab_list = [(sp.id_to_piece(i), sp.get_score(i)) for i in range(sp.get_piece_size())]
    return XLMRobertaTokenizer(vocab=vocab_list)


@pytest.fixture
def tiny_encoder(tiny_tokenizer: XLMRobertaTokenizer) -> XLMRobertaModel:
    """A small, randomly-initialized XLM-R-architecture encoder matching the
    tiny tokenizer's vocab -- stands in for xlm-roberta-large."""
    config = XLMRobertaConfig(
        vocab_size=tiny_tokenizer.vocab_size,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=300,
        pad_token_id=tiny_tokenizer.pad_token_id,
    )
    return XLMRobertaModel(config)


# ===========================================================================
# Label space tests
# ===========================================================================


def test_department_class_count_matches_locked_taxonomy() -> None:
    """
    The specialist head MUST have 13 classes, matching
    src/specialty_mapping.py's validated 13-department schema -- NOT a
    placeholder count like 10. This test is a hard guard against the head
    size silently drifting out of sync with the actual label space the
    dataset was built against.
    """
    assert tdm.NUM_DEPARTMENT_CLASSES == 13
    assert len(tdm.DEPARTMENT_CODES) == 13
    assert len(set(tdm.DEPARTMENT_CODES)) == 13  # no duplicates


def test_severity_class_count_is_five() -> None:
    assert tdm.NUM_SEVERITY_CLASSES == 5
    assert tdm.SEVERITY_LABELS == ["S1", "S2", "S3", "S4", "S5"]


def test_department_codes_match_specialty_mapping_module() -> None:
    """Catches drift if specialty_mapping.py's DEPARTMENTS dict is ever
    edited without updating this script's label space derivation."""
    from specialty_mapping import DEPARTMENTS

    assert set(tdm.DEPARTMENT_CODES) == set(DEPARTMENTS.keys())


# ===========================================================================
# Vocabulary injection tests
# ===========================================================================


def test_vocab_injection_plan_derives_from_real_perturbation_table() -> None:
    """
    The injection plan must be built FROM hinglish_perturbation.py's actual
    variant table, not a hardcoded guess list. We check this by cross-
    referencing against the module directly.
    """
    from hinglish_perturbation import _VARIANT_TABLE

    plan = tdm.build_vocab_injection_plan()
    assert len(plan.canonical_to_variants) >= len(_VARIANT_TABLE)
    assert plan.n_new_tokens > 0
    # spot-check a known canonical/variant pair from the real table
    assert "dard" in plan.canonical_to_variants
    assert "dardh" in plan.canonical_to_variants["dard"]


def test_no_fabricated_placeholder_tokens_in_injection_plan() -> None:
    """
    Guards against accidentally reintroducing the originally-requested but
    nonexistent placeholder tokens ("drd", "bkar", "shans") that don't
    correspond to anything in our actual perturbation engine or dataset.
    """
    plan = tdm.build_vocab_injection_plan()
    forbidden_tokens = {"drd", "bkar", "shans"}
    assert forbidden_tokens.isdisjoint(set(plan.new_tokens)), (
        "Found fabricated placeholder token(s) not derived from "
        "hinglish_perturbation.py's real variant table."
    )


def test_vocab_injection_resizes_embeddings_correctly(
    tiny_tokenizer: XLMRobertaTokenizer, tiny_encoder: XLMRobertaModel
) -> None:
    vocab_size_before = len(tiny_tokenizer)
    embedding_rows_before = tiny_encoder.get_input_embeddings().weight.shape[0]
    assert vocab_size_before == embedding_rows_before

    plan = tdm.build_vocab_injection_plan()
    n_added = tdm.inject_vocabulary_and_init_embeddings(tiny_encoder, tiny_tokenizer, plan)

    vocab_size_after = len(tiny_tokenizer)
    embedding_rows_after = tiny_encoder.get_input_embeddings().weight.shape[0]

    assert n_added > 0
    assert vocab_size_after == vocab_size_before + n_added
    assert embedding_rows_after == vocab_size_after, (
        "Embedding matrix row count must exactly match tokenizer length "
        "after injection -- a mismatch here means resize_token_embeddings "
        "and add_tokens went out of sync."
    )


def test_canonical_embedding_initialization_is_exact_copy(
    tiny_tokenizer: XLMRobertaTokenizer, tiny_encoder: XLMRobertaModel
) -> None:
    """
    Verifies W[t_new] = W[t_anchor] exactly (before any training has
    occurred) for genuinely NEW tokens, including the multi-subword-anchor
    fallback case (mean of subword embeddings) when the canonical word isn't
    a single token in this tokenizer's vocabulary.

    Also verifies the corrected behavior for a real edge case found during
    testing: a "variant" string that already existed as an ordinary
    pre-trained base-vocabulary token (e.g. "ap" is both a real subword
    piece in our tiny tokenizer AND listed as a phonetic variant of "aap")
    must NOT have its existing embedding overwritten -- canonical-anchor
    init only applies to genuinely new, previously-nonexistent tokens.
    """
    vocab_size_before = len(tiny_tokenizer)
    embedding_layer = tiny_encoder.get_input_embeddings()
    # snapshot embeddings that already exist before injection, to verify
    # any of them that happen to be listed as "variants" are left untouched
    pre_existing_snapshot = embedding_layer.weight.detach().clone()

    plan = tdm.build_vocab_injection_plan()

    # IMPORTANT: compute each anchor's expected vector BEFORE injection,
    # exactly as inject_vocabulary_and_init_embeddings itself does internally
    # for each (canonical_word, variants) pair, one at a time, before adding
    # that pair's variant tokens to the vocabulary. We must NOT recompute
    # anchor vectors by re-tokenizing the canonical word AFTER injection --
    # a real effect found during testing is that adding a variant token can
    # itself change how the canonical word re-tokenizes (e.g. once "apka" is
    # in-vocab, "aapka" may now greedily tokenize as a different subword
    # split than it did before "apka" existed), which would make a
    # post-injection recomputation circular and not actually verify what the
    # injection code computed at the time it ran.
    expected_anchor_vectors: dict[str, torch.Tensor] = {}
    for canonical_word in plan.canonical_to_variants:
        anchor_ids = tiny_tokenizer.encode(canonical_word, add_special_tokens=False)
        if len(anchor_ids) == 1:
            expected_anchor_vectors[canonical_word] = embedding_layer.weight[anchor_ids[0]].clone()
        else:
            expected_anchor_vectors[canonical_word] = (
                embedding_layer.weight[anchor_ids].mean(dim=0).clone()
            )

    tdm.inject_vocabulary_and_init_embeddings(tiny_encoder, tiny_tokenizer, plan)
    embedding_layer = tiny_encoder.get_input_embeddings()  # re-fetch post-resize

    # A variant token can legitimately be listed under more than one
    # canonical anchor (e.g. "nai" is a documented variant of both "nahi"
    # and "nahin"). The production code resolves this deterministically:
    # whichever anchor is encountered FIRST in dict-iteration order wins,
    # and later claims on the same token id are skipped. We replicate that
    # same first-wins resolution here so the test's expectation matches what
    # the implementation actually (and intentionally) does, rather than
    # assuming every listed anchor independently controls its variants.
    first_owner_anchor_vec: dict[int, torch.Tensor] = {}
    for canonical_word, variants in plan.canonical_to_variants.items():
        for variant_token in variants:
            variant_id = tiny_tokenizer.convert_tokens_to_ids(variant_token)
            if variant_id is None or variant_id == tiny_tokenizer.unk_token_id:
                continue
            if variant_id >= vocab_size_before and variant_id not in first_owner_anchor_vec:
                first_owner_anchor_vec[variant_id] = expected_anchor_vectors[canonical_word]

    checked_new_pairs = 0
    checked_preexisting_preserved = 0
    for canonical_word, variants in plan.canonical_to_variants.items():
        for variant_token in variants:
            variant_id = tiny_tokenizer.convert_tokens_to_ids(variant_token)
            if variant_id is None or variant_id == tiny_tokenizer.unk_token_id:
                continue
            variant_vec = embedding_layer.weight[variant_id]

            if variant_id < vocab_size_before:
                # Pre-existing token: must be UNCHANGED from before injection,
                # not overwritten with the anchor's vector.
                assert torch.allclose(variant_vec, pre_existing_snapshot[variant_id]), (
                    f"Pre-existing token {variant_token!r} (id={variant_id}) "
                    f"was incorrectly overwritten by canonical-anchor init "
                    f"for {canonical_word!r} -- this destroys real "
                    f"pretrained signal and must not happen."
                )
                checked_preexisting_preserved += 1
            else:
                # Genuinely new token: must exactly match whichever anchor
                # was the first to claim it (see first_owner_anchor_vec note
                # above), not necessarily THIS canonical_word's anchor if
                # another anchor claimed it earlier in iteration order.
                expected_vec = first_owner_anchor_vec[variant_id]
                assert torch.allclose(expected_vec, variant_vec), (
                    f"Canonical-anchor copy mismatch for {canonical_word!r} -> "
                    f"{variant_token!r} (expected first-claiming anchor's vector)"
                )
                checked_new_pairs += 1

    assert checked_new_pairs > 0, "No new canonical/variant pairs were actually checked"
    assert checked_preexisting_preserved > 0, (
        "Expected at least one variant token to already exist in the base "
        "vocabulary in this test fixture (e.g. 'ap') -- if this assertion "
        "fails, the test fixture's vocabulary changed and this regression "
        "guard may no longer be exercising the pre-existing-token-skip path."
    )


def test_vocab_injection_is_idempotent_on_already_added_tokens(
    tiny_tokenizer: XLMRobertaTokenizer, tiny_encoder: XLMRobertaModel
) -> None:
    """Calling injection twice should not double-add or corrupt the vocab."""
    plan = tdm.build_vocab_injection_plan()
    n_added_first = tdm.inject_vocabulary_and_init_embeddings(tiny_encoder, tiny_tokenizer, plan)
    vocab_size_after_first = len(tiny_tokenizer)

    n_added_second = tdm.inject_vocabulary_and_init_embeddings(tiny_encoder, tiny_tokenizer, plan)
    vocab_size_after_second = len(tiny_tokenizer)

    assert n_added_first > 0
    assert n_added_second == 0, "Re-adding the same tokens should add zero new tokens"
    assert vocab_size_after_second == vocab_size_after_first


# ===========================================================================
# Model architecture tests
# ===========================================================================


def test_model_forward_pass_output_shapes(
    tiny_tokenizer: XLMRobertaTokenizer, tiny_encoder: XLMRobertaModel
) -> None:
    plan = tdm.build_vocab_injection_plan()
    tdm.inject_vocabulary_and_init_embeddings(tiny_encoder, tiny_tokenizer, plan)
    model = tdm.MediTriageTransformer(tiny_encoder)

    texts = ["mera bahut dard ho raha hai", "routine follow up no acute distress"]
    encoding = tiny_tokenizer(texts, padding=True, truncation=True, max_length=32, return_tensors="pt")

    specialist_logits, severity_logits = model(encoding["input_ids"], encoding["attention_mask"])

    assert specialist_logits.shape == (2, tdm.NUM_DEPARTMENT_CLASSES)
    assert severity_logits.shape == (2, tdm.NUM_SEVERITY_CLASSES)


def test_model_gradients_flow_on_backward(
    tiny_tokenizer: XLMRobertaTokenizer, tiny_encoder: XLMRobertaModel
) -> None:
    plan = tdm.build_vocab_injection_plan()
    tdm.inject_vocabulary_and_init_embeddings(tiny_encoder, tiny_tokenizer, plan)
    model = tdm.MediTriageTransformer(tiny_encoder)

    encoding = tiny_tokenizer(["mera dard bahut zyada hai"], return_tensors="pt")
    specialist_logits, severity_logits = model(encoding["input_ids"], encoding["attention_mask"])

    loss_fn = tdm.JointLoss(tdm.JointLossWeights())
    loss_dict = loss_fn(
        specialist_logits, torch.tensor([3]), severity_logits, torch.tensor([1])
    )
    loss_dict["joint_loss"].backward()

    has_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0 for p in model.parameters()
    )
    assert has_grad, "No gradients flowed through the model on backward()"


# ===========================================================================
# Joint loss tests
# ===========================================================================


def test_joint_loss_weighting_formula_is_correct() -> None:
    """L_joint = alpha * L_specialist + beta * L_severity, with the
    project's baseline alpha=1.0, beta=1.2."""
    weights = tdm.JointLossWeights(alpha_specialist=1.0, beta_severity=1.2)
    loss_fn = tdm.JointLoss(weights)

    torch.manual_seed(0)
    specialist_logits = torch.randn(4, tdm.NUM_DEPARTMENT_CLASSES, requires_grad=True)
    severity_logits = torch.randn(4, tdm.NUM_SEVERITY_CLASSES, requires_grad=True)
    specialist_labels = torch.tensor([0, 3, 7, 12])
    severity_labels = torch.tensor([0, 1, 2, 4])

    loss_dict = loss_fn(specialist_logits, specialist_labels, severity_logits, severity_labels)

    expected_joint = (
        1.0 * loss_dict["specialist_loss"].item() + 1.2 * loss_dict["severity_loss"].item()
    )
    assert abs(expected_joint - loss_dict["joint_loss"].item()) < 1e-5


def test_joint_loss_default_weights_match_spec() -> None:
    weights = tdm.JointLossWeights()
    assert weights.alpha_specialist == 1.0
    assert weights.beta_severity == 1.2


# ===========================================================================
# Scheduler tests
# ===========================================================================


def test_linear_warmup_schedule_ramps_then_decays() -> None:
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup

    dummy_param = [torch.nn.Parameter(torch.randn(2, 2))]
    optimizer = AdamW(dummy_param, lr=2e-5)
    total_steps = 100
    warmup_steps = int(total_steps * 0.10)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    lrs = []
    for _ in range(total_steps):
        optimizer.step()
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])

    assert warmup_steps == 10
    assert lrs[0] < lrs[warmup_steps - 1], "LR should increase during warmup"
    assert lrs[-1] < lrs[warmup_steps], "LR should decay after warmup ends"
    assert lrs[-1] == pytest.approx(0.0, abs=1e-8)


# ===========================================================================
# Dataset loading tests (the NaN-text bug found during integration testing)
# ===========================================================================


def test_load_split_rows_drops_rows_with_missing_text(tmp_path: Path) -> None:
    """
    Regression test for a real bug found during integration testing: 33
    MTSamples seed rows have a missing transcription, which serializes to
    NaN when read back from the processed CSV and crashes the tokenizer if
    not filtered. load_split_rows must drop these rows rather than crash or
    silently pass NaN through to the tokenizer.
    """
    import pandas as pd

    csv_path = tmp_path / "fake_dataset.csv"
    df = pd.DataFrame(
        {
            "tracking_id": ["a::v0::1", "b::v0::2", "c::v0::3"],
            "seed_id": ["a", "b", "c"],
            "text": ["valid clinical text here", None, "another valid row"],
            "department_code": ["GEN_MED", "GEN_MED", "SURGERY"],
            "severity_heuristic": ["S4", "S4", "S3"],
            "split": ["train", "train", "train"],
        }
    )
    df.to_csv(csv_path, index=False)

    rows = tdm.load_split_rows(csv_path, "train")

    assert len(rows) == 2, "Expected the NaN-text row to be dropped"
    assert all(isinstance(r["text"], str) and len(r["text"].strip()) > 0 for r in rows)


def test_load_split_rows_raises_on_missing_split(tmp_path: Path) -> None:
    import pandas as pd

    csv_path = tmp_path / "fake_dataset.csv"
    pd.DataFrame(
        {
            "tracking_id": ["a::v0::1"],
            "seed_id": ["a"],
            "text": ["valid text"],
            "department_code": ["GEN_MED"],
            "severity_heuristic": ["S4"],
            "split": ["train"],
        }
    ).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="not found"):
        tdm.load_split_rows(csv_path, "val")


# ===========================================================================
# Running metrics tests
# ===========================================================================


def test_running_metrics_accuracy_computation() -> None:
    metrics = tdm.RunningMetrics()

    specialist_logits = torch.tensor([[5.0, 0.0], [0.0, 5.0]])  # predicts class 0, then 1
    severity_logits = torch.tensor([[5.0, 0.0], [5.0, 0.0]])  # predicts class 0 both times
    specialist_labels = torch.tensor([0, 1])  # both correct
    severity_labels = torch.tensor([0, 1])  # first correct, second wrong

    metrics.update(
        specialist_loss=1.0,
        severity_loss=2.0,
        specialist_logits=specialist_logits,
        severity_logits=severity_logits,
        specialist_labels=specialist_labels,
        severity_labels=severity_labels,
    )

    assert metrics.specialist_acc == pytest.approx(1.0)
    assert metrics.severity_acc == pytest.approx(0.5)
    assert metrics.specialist_avg_loss == pytest.approx(1.0)
    assert metrics.severity_avg_loss == pytest.approx(2.0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
