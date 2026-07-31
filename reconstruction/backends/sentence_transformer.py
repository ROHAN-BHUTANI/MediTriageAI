"""SentenceTransformer Cluster Backend (optional).

Uses a pretrained SentenceTransformer model for dense embeddings and
MiniBatchKMeans for clustering.  Requires `sentence-transformers` to
be installed; gracefully raises ImportError if missing.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from reconstruction.backends import ClusterBackend


class SentenceTransformerBackend(ClusterBackend):
    """Dense embedding backend using SentenceTransformers.

    Args:
        model_name: HuggingFace model identifier
                     (default: all-MiniLM-L6-v2).
        batch_size: Encoding batch size.
        device: Torch device string (e.g. "cpu", "cuda").
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 256,
        device: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._device = device
        self._model: Any = None

    def _load_model(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "SentenceTransformer backend requires `sentence-transformers`. "
                "Install it with: pip install sentence-transformers"
            ) from e
        self._model = SentenceTransformer(self._model_name, device=self._device)

    def fit(self, texts: list[str], **kwargs: Any) -> None:
        # Pretrained model – nothing to fit, just ensure loaded
        if self._model is None:
            self._load_model()

    def encode(self, texts: list[str], **kwargs: Any) -> np.ndarray:
        if self._model is None:
            self._load_model()
        embeddings = self._model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def cluster(
        self,
        features: np.ndarray,
        n_clusters: int,
        random_state: int = 42,
    ) -> np.ndarray:
        n = features.shape[0]
        if n_clusters <= 1 or n <= 1:
            return np.zeros(n, dtype=np.int32)

        km = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            batch_size=min(1024, n),
            n_init=3,
        )
        return km.fit_predict(features).astype(np.int32)

    @property
    def name(self) -> str:
        return f"SentenceTransformer({self._model_name})"
