import time
from pathlib import Path
from typing import Any, Callable
import torch
import torch.nn as nn

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
    if any("bert" in k for k in keys) and not any("albert" in k or "distilbert" in k or "roberta" in k for k in keys):
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
    extra_states: dict | None = None
) -> None:
    """Save a versioned and metadata-rich checkpoint dict."""
    checkpoint = {
        "version": "2.0",
        "model_short_name": model_short_name,
        "backbone_name": backbone_name,
        "config": config,
        "state_dict": state_dict,
        "model_state_dict": state_dict  # Alias for legacy loaders reading from disk
    }
    if extra_states:
        checkpoint.update(extra_states)
    torch.save(checkpoint, path)

def load_checkpoint(path: Path, map_location="cpu") -> dict:
    """Load a checkpoint, dynamically wrapping raw legacy state dicts in a unified format."""
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file not found at: {path}")
        
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    
    # 1. Determine if this checkpoint lacks version 2.0 metadata
    is_legacy = False
    if not isinstance(checkpoint, dict):
        is_legacy = True
    elif "version" not in checkpoint or "state_dict" not in checkpoint:
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
            "model_state_dict": state_dict  # Alias for backward compatibility
        }
        
        # Preserve training states if present
        if isinstance(checkpoint, dict):
            for k in ("optimizer_state_dict", "scheduler_state_dict", "scaler_state_dict", 
                      "epoch", "history", "best_val_loss", "patience_counter", "metadata"):
                if k in checkpoint:
                    unified_checkpoint[k] = checkpoint[k]
        return unified_checkpoint
        
    # For modern checkpoints, ensure model_state_dict alias is present
    if "model_state_dict" not in checkpoint and "state_dict" in checkpoint:
        checkpoint["model_state_dict"] = checkpoint["state_dict"]
    return checkpoint

def reconstruct_model_and_tokenizer(checkpoint_dict: dict, device="cpu") -> tuple[nn.Module, Any, Any]:
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
    for k in state_dict.keys():
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
        config = AutoConfig.from_pretrained(checkpoint_dict["backbone_name"], local_files_only=False)
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
            config = AlbertConfig(hidden_size=hidden_size, num_hidden_layers=num_hidden_layers, vocab_size=vocab_size)
        elif model_short_name in ("distil_bert", "distilbert_multi"):
            from transformers import DistilBertConfig
            config = DistilBertConfig(dim=hidden_size, n_layers=num_hidden_layers, vocab_size=vocab_size)
        elif model_short_name == "mbert":
            from transformers import BertConfig
            config = BertConfig(hidden_size=hidden_size, num_hidden_layers=num_hidden_layers, vocab_size=vocab_size)
        else:
            from transformers import XLMRobertaConfig
            config = XLMRobertaConfig(hidden_size=hidden_size, num_hidden_layers=num_hidden_layers, vocab_size=vocab_size)
            
    # Build model wrapper
    if model_short_name == "emergent_path_triage":
        from models.emergent_path_triage.config import EmergentPathTriageConfig
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
