"""TF-IDF Cluster Backend (default fallback).

Uses sklearn TfidfVectorizer for encoding and MiniBatchKMeans for
clustering.  Zero external dependencies beyond scikit-learn.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from reconstruction.backends import ClusterBackend


class TfidfBackend(ClusterBackend):
    """TF-IDF + MiniBatchKMeans backend.

    Args:
        max_features: Maximum vocabulary size for the TF-IDF vectorizer.
        ngram_range: Tuple of (min_n, max_n) for n-gram extraction.
    """

    def __init__(
        self,
        max_features: int = 10_000,
        ngram_range: tuple[int, int] = (1, 3),
    ) -> None:
        self._max_features = max_features
        self._ngram_range = ngram_range
        self._vectorizer: TfidfVectorizer | None = None

    def fit(self, texts: list[str], **kwargs: Any) -> None:
        n = len(texts)
        self._vectorizer = TfidfVectorizer(
            max_features=self._max_features,
            ngram_range=self._ngram_range,
            sublinear_tf=True,
            strip_accents="unicode",
            min_df=2 if n > 100 else 1,
        )
        self._vectorizer.fit(texts)

    def encode(self, texts: list[str], **kwargs: Any) -> np.ndarray:
        if self._vectorizer is None:
            raise RuntimeError("TfidfBackend.fit() must be called before encode().")
        # Returns a sparse matrix; convert to dense for uniform interface
        sparse = self._vectorizer.transform(texts)
        return sparse.toarray()

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
        return "TF-IDF"
