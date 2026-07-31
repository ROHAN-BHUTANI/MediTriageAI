"""Cluster Backend – Abstract Interface.

Defines the ClusterBackend protocol that all embedding/clustering
implementations must satisfy.  Stage 3 interacts exclusively with
this interface and never knows which backend is active.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class ClusterBackend(ABC):
    """Abstract base class for clinical phenotype clustering backends.

    Every backend must implement three methods:
      1. fit()    – learn representations from a corpus of texts.
      2. encode() – transform texts into a numeric feature matrix.
      3. cluster() – assign cluster IDs to the encoded representations.

    The reconstruction engine calls them in sequence:
        backend.fit(texts)
        features = backend.encode(texts)
        labels   = backend.cluster(features, n_clusters)
    """

    @abstractmethod
    def fit(self, texts: list[str], **kwargs: Any) -> None:
        """Learn internal representations from the corpus.

        For TF-IDF this fits the vectorizer; for neural models this
        may be a no-op (pretrained) or a fine-tuning step.

        Args:
            texts: All texts for a single department.
            **kwargs: Backend-specific parameters.
        """

    @abstractmethod
    def encode(self, texts: list[str], **kwargs: Any) -> np.ndarray:
        """Encode texts into a 2-D feature matrix.

        Args:
            texts: Texts to encode (same or subset of fit corpus).
            **kwargs: Backend-specific parameters.

        Returns:
            np.ndarray of shape (n_samples, n_features).
        """

    @abstractmethod
    def cluster(
        self,
        features: np.ndarray,
        n_clusters: int,
        random_state: int = 42,
    ) -> np.ndarray:
        """Assign cluster IDs to feature vectors.

        Args:
            features: Output of encode(), shape (n_samples, n_features).
            n_clusters: Desired number of clusters.
            random_state: Seed for reproducibility.

        Returns:
            np.ndarray of int32 cluster labels, shape (n_samples,).
        """

    @property
    def name(self) -> str:
        """Human-readable backend name for logging and reports."""
        return self.__class__.__name__
