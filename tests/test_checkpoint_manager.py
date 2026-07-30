import torch

from src.checkpoint_manager import (
    detect_model_class_from_state_dict,
    load_checkpoint,
    save_checkpoint,
)


def test_save_and_load_checkpoint_v2(tmp_path):
    checkpoint_path = tmp_path / "model_v2.pt"

    # Mock model parameters and config
    config = {"encoder_lr": 2e-5, "classifier_lr": 1e-4, "batch_size": 8}
    state_dict = {
        "encoder.embeddings.word_embeddings.weight": torch.randn(10, 64),
        "classifier_specialist.weight": torch.randn(13, 64),
        "classifier_severity.weight": torch.randn(5, 64),
    }
    extra = {"epoch": 2, "best_val_loss": 0.123}

    # Save checkpoint
    save_checkpoint(
        path=checkpoint_path,
        model_short_name="indic_bert",
        backbone_name="ai4bharat/indic-bert",
        config=config,
        state_dict=state_dict,
        extra_states=extra,
    )

    # Assert file exists
    assert checkpoint_path.exists()

    # Load checkpoint
    loaded = load_checkpoint(checkpoint_path)

    assert loaded["version"] == "3.0"
    assert loaded["model_short_name"] == "indic_bert"
    assert loaded["backbone_name"] == "ai4bharat/indic-bert"
    assert loaded["config"] == config
    assert loaded["epoch"] == 2
    assert loaded["best_val_loss"] == 0.123
    assert torch.equal(
        loaded["state_dict"]["classifier_specialist.weight"],
        state_dict["classifier_specialist.weight"],
    )
    # Backward compatibility key check
    assert "model_state_dict" in loaded
    assert torch.equal(
        loaded["model_state_dict"]["classifier_specialist.weight"],
        state_dict["classifier_specialist.weight"],
    )


def test_detect_model_class_legacy_patterns():
    # 1. IndicBERT (contains albert)
    indic_state = {
        "encoder.albert.embeddings.word_embeddings.weight": torch.randn(100, 64)
    }
    assert detect_model_class_from_state_dict(indic_state) == "IndicBertModel"

    # 2. DistilBERT (contains distilbert)
    distil_state = {
        "encoder.distilbert.embeddings.word_embeddings.weight": torch.randn(100, 64)
    }
    assert (
        detect_model_class_from_state_dict(distil_state)
        == "DistilBertMultilingualModel"
    )

    # 3. mBERT (contains bert but not albert/distilbert)
    mbert_state = {
        "encoder.bert.embeddings.word_embeddings.weight": torch.randn(100, 64)
    }
    assert detect_model_class_from_state_dict(mbert_state) == "MBertModel"

    # 4. XLMRobertaLargeModel (contains roberta, hidden size > 1000)
    xlmr_large_state = {
        "encoder.roberta.embeddings.word_embeddings.weight": torch.randn(100, 1024)
    }
    assert (
        detect_model_class_from_state_dict(xlmr_large_state) == "XLMRobertaLargeModel"
    )

    # 5. EmergentPathTriageModel (contains roberta, hidden size <= 1000)
    epath_state = {
        "encoder.roberta.embeddings.word_embeddings.weight": torch.randn(100, 768)
    }
    assert detect_model_class_from_state_dict(epath_state) == "EmergentPathTriageModel"


def test_load_legacy_raw_checkpoint(tmp_path):
    checkpoint_path = tmp_path / "legacy.pt"

    # Create raw state dict checkpoint (representing IndicBERT)
    legacy_state_dict = {
        "encoder.albert.embeddings.word_embeddings.weight": torch.randn(10, 64),
        "classifier_specialist.weight": torch.randn(13, 64),
    }
    torch.save(legacy_state_dict, checkpoint_path)

    loaded = load_checkpoint(checkpoint_path)

    assert loaded["version"] == "1.0-legacy"
    assert loaded["model_short_name"] == "indic_bert"
    assert loaded["backbone_name"] == "ai4bharat/indic-bert"
    assert loaded["config"] == {}
    assert "state_dict" in loaded
    assert "model_state_dict" in loaded
    assert torch.equal(
        loaded["state_dict"]["classifier_specialist.weight"],
        legacy_state_dict["classifier_specialist.weight"],
    )
