"""Backend Factory.

Resolves a backend name string from config to a concrete ClusterBackend
instance.  Keeps the rest of the engine decoupled from backend classes.
"""

from __future__ import annotations

import logging

from reconstruction.backends import ClusterBackend
from reconstruction.config import ReconstructionConfig

logger = logging.getLogger(__name__)


def create_backend(cfg: ReconstructionConfig) -> ClusterBackend:
    """Instantiate the clustering backend specified in config.

    Args:
        cfg: Reconstruction configuration (reads ``embedding_model``).

    Returns:
        A concrete ClusterBackend instance.

    Raises:
        ValueError: If the backend name is unrecognised.
    """
    name = cfg.embedding_model.lower()

    if name == "tfidf":
        from reconstruction.backends.tfidf import TfidfBackend

        backend = TfidfBackend(
            max_features=cfg.tfidf_max_features,
            ngram_range=(1, 3),
        )
        logger.info("Using TF-IDF cluster backend")
        return backend

    if name == "sentence_transformer":
        from reconstruction.backends.sentence_transformer import SentenceTransformerBackend

        backend = SentenceTransformerBackend(
            model_name=cfg.sentence_transformer_name,
        )
        logger.info(
            "Using SentenceTransformer cluster backend: %s",
            cfg.sentence_transformer_name,
        )
        return backend

    raise ValueError(
        f"Unknown embedding_model '{cfg.embedding_model}'. "
        f"Supported: tfidf, sentence_transformer"
    )
