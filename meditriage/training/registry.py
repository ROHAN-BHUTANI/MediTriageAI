"""Registry for Model Backbones, Optimizers, Schedulers, and Losses."""

from __future__ import annotations

# Registered Transformer Backbones
BACKBONE_REGISTRY: dict[str, str] = {
    "xlm-roberta-base": "xlm-roberta-base",
    "xlm-roberta-large": "xlm-roberta-large",
    "muril": "google/muril-base-cased",
    "indicbert": "ai4bharat/indic-bert",
    "distilbert-multilingual": "distilbert-base-multilingual-cased",
}

# Registered Optimizers
OPTIMIZER_REGISTRY: list[str] = ["adamw", "sgd", "adam"]

# Registered Schedulers
SCHEDULER_REGISTRY: list[str] = ["cosine", "linear", "onecycle", "reducelronplateau"]

# Registered Loss Functions
LOSS_REGISTRY: list[str] = ["cross_entropy", "weighted_cross_entropy", "focal"]


def get_backbone_model_id(name_or_key: str) -> str:
    """Resolve backbone model name or shortcut key to HuggingFace model ID."""
    key = name_or_key.lower()
    return BACKBONE_REGISTRY.get(key, name_or_key)
