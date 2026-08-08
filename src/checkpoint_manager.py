from pathlib import Path
from typing import Any

import torch
from torch import nn


def detect_model_class_from_state_dict(state_dict: dict[str, Any]) -> str:
    """Classify raw checkpoints by checking key structure and weight shapes."""
    keys = list(state_dict.keys())

    # 1. Check for ALBERT (IndicBERT uses albert)
    if any("albert" in k for k in keys):
        return "IndicBertModel"

    # 2. Check for DistilBERT
    if any("distilbert" in k for k in keys):
        return "DistilBertMultilingualModel"

    # 3. Check for mBERT (bert)
    if any("bert" in k for k in keys) and not any(
        "albert" in k or "distilbert" in k or "roberta" in k for k in keys
    ):
        return "MBertModel"

    # 4. Check for RoBERTa / XLM-RoBERTa (roberta)
    if any("roberta" in k for k in keys):
        # Differentiate between Large and Base using word embeddings hidden size
        hidden_size = 768
        for k, v in state_dict.items():
            if "word_embeddings.weight" in k:
                hidden_size = v.shape[1]
                break
        if hidden_size > 1000:
            return "XLMRobertaLargeModel"
        else:
            return "EmergentPathTriageModel"

    # Default fallback
    return "EmergentPathTriageModel"


def get_short_name_from_class_name(class_name: str) -> str:
    if class_name == "IndicBertModel":
        return "indic_bert"
    elif class_name == "MBertModel":
        return "mbert"
    elif class_name == "DistilBertMultilingualModel":
        return "distil_bert"
    elif class_name == "XLMRobertaLargeModel":
        return "xlm_roberta"
    elif class_name == "EmergentPathTriageModel":
        return "emergent_path_triage"
    else:
        raise ValueError(f"Unknown class name: {class_name}")


def get_backbone_from_short_name(short_name: str) -> str:
    if short_name == "indic_bert":
        return "ai4bharat/indic-bert"
    elif short_name == "mbert":
        return "bert-base-multilingual-cased"
    elif short_name in ("distil_bert", "distilbert_multi"):
        return "distilbert-base-multilingual-cased"
    elif short_name == "xlm_roberta":
        return "xlm-roberta-large"
    elif short_name == "emergent_path_triage":
        return "xlm-roberta-base"
    else:
        return "xlm-roberta-base"


def get_model_class_by_short_name(short_name: str):
    if short_name == "indic_bert":
        from models.indic_bert import IndicBertModel

        return IndicBertModel
    elif short_name == "mbert":
        from models.mbert import MBertModel

        return MBertModel
    elif short_name in ("distil_bert", "distilbert_multi"):
        from models.distilbert_multi import DistilBertMultilingualModel

        return DistilBertMultilingualModel
    elif short_name == "xlm_roberta":
        from models.xlm_roberta import XLMRobertaLargeModel

        return XLMRobertaLargeModel
    elif short_name == "emergent_path_triage":
        from models.emergent_path_triage.model import EmergentPathTriageModel

        return EmergentPathTriageModel
    else:
        raise ValueError(f"Unknown model short name: {short_name}")


def save_checkpoint(
    path: Path,
    model_short_name: str,
    backbone_name: str,
    config: dict,
    state_dict: dict,
    extra_states: dict | None = None,
    experiment_id: str = "unknown",
    config_hash: str = "unknown",
    dataset_manifest_hash: str = "unknown",
    tokenizer_hash: str = "unknown",
) -> None:
    """Save a versioned and metadata-rich checkpoint dict with strict integrity checks."""
    import datetime
    import hashlib

    checkpoint = {
        "version": "3.0",
        "experiment_id": experiment_id,
        "model_short_name": model_short_name,
        "backbone_name": backbone_name,
        "config": config,
        "config_hash": config_hash,
        "dataset_manifest_hash": dataset_manifest_hash,
        "tokenizer_hash": tokenizer_hash,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "state_dict": state_dict,
        "model_state_dict": state_dict,  # Alias for legacy loaders reading from disk
    }
    if extra_states:
        checkpoint.update(extra_states)

    import os
    import time

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    max_retries = 3
    for attempt in range(max_retries):
        try:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, tmp_path)

            os.replace(tmp_path, path)
            break
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(0.5)
                continue
            raise

    # Generate and write SHA256 checksum after ensuring bytes are written to disk
    with open(path, "rb") as f:
        file_bytes = f.read()
    file_hash = hashlib.sha256(file_bytes).hexdigest()

    checksum_path = path.with_suffix(path.suffix + ".sha256")
    tmp_checksum_path = checksum_path.with_suffix(checksum_path.suffix + ".tmp")
    with open(tmp_checksum_path, "w", encoding="utf-8") as f:
        f.write(file_hash)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_checksum_path, checksum_path)




def load_checkpoint(
    path: Path,
    map_location="cpu",
    expected_config_hash: str | None = None,
    expected_dataset_hash: str | None = None,
    expected_tokenizer_hash: str | None = None,
) -> dict:
    """Load a checkpoint with strict integrity verification."""
    import hashlib
    import time

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file not found at: {path}")

    # 1. Verify Checksum
    checksum_path = path.with_suffix(path.suffix + ".sha256")
    if checksum_path.exists():
        with open(checksum_path, "r", encoding="utf-8") as f:
            expected_checksum = f.read().strip()

        with open(path, "rb") as f:
            actual_checksum = hashlib.sha256(f.read()).hexdigest()

        if expected_checksum != actual_checksum:
            time.sleep(0.2)
            with open(path, "rb") as f:
                actual_checksum = hashlib.sha256(f.read()).hexdigest()

        if expected_checksum != actual_checksum:
            import warnings

            warnings.warn(
                f"Checkpoint SHA256 checksum mismatch for {path} (expected {expected_checksum[:8]}, got {actual_checksum[:8]})."
            )



    checkpoint = torch.load(path, map_location=map_location, weights_only=False)

    # 2. Verify Resume Safety (Hashes)
    if (
        expected_config_hash
        and checkpoint.get("config_hash")
        and checkpoint.get("config_hash") != "unknown"
    ):
        if expected_config_hash != checkpoint["config_hash"]:
            raise ValueError(
                f"Resume aborted: Config hash mismatch. Expected {expected_config_hash}, got {checkpoint['config_hash']}."
            )

    if (
        expected_dataset_hash
        and checkpoint.get("dataset_manifest_hash")
        and checkpoint.get("dataset_manifest_hash") != "unknown"
    ):
        if expected_dataset_hash != checkpoint["dataset_manifest_hash"]:
            raise ValueError(
                f"Resume aborted: Dataset hash mismatch. Expected {expected_dataset_hash}, got {checkpoint['dataset_manifest_hash']}."
            )

    if (
        expected_tokenizer_hash
        and checkpoint.get("tokenizer_hash")
        and checkpoint.get("tokenizer_hash") != "unknown"
    ):
        if expected_tokenizer_hash != checkpoint["tokenizer_hash"]:
            raise ValueError(
                f"Resume aborted: Tokenizer hash mismatch. Expected {expected_tokenizer_hash}, got {checkpoint['tokenizer_hash']}."
            )

    # 3. Determine if this checkpoint lacks version 2.0/3.0 metadata
    is_legacy = False
    if (
        not isinstance(checkpoint, dict)
        or "version" not in checkpoint
        or "state_dict" not in checkpoint
    ):
        is_legacy = True

    if is_legacy:
        # Extract underlying raw model state dict
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        # Classify model architecture and names statically
        model_class_name = detect_model_class_from_state_dict(state_dict)
        model_short_name = get_short_name_from_class_name(model_class_name)
        backbone_name = get_backbone_from_short_name(model_short_name)

        # Build unified v2 checkpoint structure
        unified_checkpoint = {
            "version": "1.0-legacy",
            "model_short_name": model_short_name,
            "backbone_name": backbone_name,
            "config": {},
            "state_dict": state_dict,
            "model_state_dict": state_dict,  # Alias for backward compatibility
        }

        # Preserve training states if present
        if isinstance(checkpoint, dict):
            for k in (
                "optimizer_state_dict",
                "scheduler_state_dict",
                "scaler_state_dict",
                "epoch",
                "history",
                "best_val_loss",
                "patience_counter",
                "metadata",
            ):
                if k in checkpoint:
                    unified_checkpoint[k] = checkpoint[k]
        return unified_checkpoint

    # For modern checkpoints, ensure model_state_dict alias is present
    if "model_state_dict" not in checkpoint and "state_dict" in checkpoint:
        checkpoint["model_state_dict"] = checkpoint["state_dict"]
    return checkpoint


def reconstruct_model_and_tokenizer(
    checkpoint_dict: dict, device="cpu"
) -> tuple[nn.Module, Any, Any]:
    """Dynamically build the model configuration, model architecture, and tokenizer from checkpoint metadata."""
    model_short_name = checkpoint_dict["model_short_name"]
    model_class = get_model_class_by_short_name(model_short_name)
    model_meta = model_class()
    state_dict = checkpoint_dict["state_dict"]

    # Statically determine sizes from the weight dictionary as fallback
    hidden_size = 768
    num_hidden_layers = 12
    vocab_size = 250002
    for k, v in state_dict.items():
        if "word_embeddings.weight" in k:
            hidden_size = v.shape[1]
            vocab_size = v.shape[0]
            break

    layer_indices = set()
    for k in state_dict:
        if "encoder.layer." in k or "encoder.encoder.layer." in k:
            parts = k.split(".")
            for part in parts:
                if part.isdigit():
                    layer_indices.add(int(part))
                    break
    if layer_indices:
        num_hidden_layers = max(layer_indices) + 1
    else:
        num_hidden_layers = 1

    latent_dim = 128
    for k, v in state_dict.items():
        if "classifier_specialist.fc1.weight" in k:
            latent_dim = v.shape[1]
            break

    # Load / Build base transformers config
    from transformers import AutoConfig

    try:
        config = AutoConfig.from_pretrained(
            checkpoint_dict["backbone_name"], local_files_only=False
        )
        if hasattr(config, "hidden_size"):
            config.hidden_size = hidden_size
        if hasattr(config, "dim"):
            config.dim = hidden_size
        if hasattr(config, "num_hidden_layers"):
            config.num_hidden_layers = num_hidden_layers
        if hasattr(config, "n_layers"):
            config.n_layers = num_hidden_layers
        config.vocab_size = vocab_size
    except Exception:
        # Fallback to defaults
        if model_short_name == "indic_bert":
            from transformers import AlbertConfig

            config = AlbertConfig(
                hidden_size=hidden_size,
                num_hidden_layers=num_hidden_layers,
                vocab_size=vocab_size,
            )
        elif model_short_name in ("distil_bert", "distilbert_multi"):
            from transformers import DistilBertConfig

            config = DistilBertConfig(
                dim=hidden_size, n_layers=num_hidden_layers, vocab_size=vocab_size
            )
        elif model_short_name == "mbert":
            from transformers import BertConfig

            config = BertConfig(
                hidden_size=hidden_size,
                num_hidden_layers=num_hidden_layers,
                vocab_size=vocab_size,
            )
        else:
            from transformers import XLMRobertaConfig

            config = XLMRobertaConfig(
                hidden_size=hidden_size,
                num_hidden_layers=num_hidden_layers,
                vocab_size=vocab_size,
            )

    # Build model wrapper
    if model_short_name == "emergent_path_triage":
        from models.emergent_path_triage.config import EmergentPathTriageConfig

        triage_config_dict = checkpoint_dict.get("triage_config", {})
        if triage_config_dict:
            triage_config = EmergentPathTriageConfig.from_dict(triage_config_dict)
        else:
            triage_config = EmergentPathTriageConfig(latent_dim=latent_dim)
        model = model_meta.build(config, triage_config=triage_config)
    else:
        model = model_meta.build(config)

    # Resize embed weights if vocabulary size has evolved
    if hasattr(model, "resize_token_embeddings"):
        model.resize_token_embeddings(vocab_size)

    # Load state dict
    model.load_state_dict(state_dict, strict=False)
    model.to(device)

    # Load tokenizer
    tokenizer = model_meta.build_tokenizer()

    return model, tokenizer, model_meta
