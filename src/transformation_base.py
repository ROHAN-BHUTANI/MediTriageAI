# src/transformation_base.py
"""Base classes for deterministic transformation plugins.
Each plugin must inherit from `TransformationPlugin` and implement:
- `name` (class attribute)
- `apply(self, text: str, rng: random.Random) -> Tuple[str, dict]`
The `apply` method returns the transformed text and a metadata dict describing the operation.
"""

import abc
import random


class TransformationPlugin(abc.ABC):
    """Abstract base class for all transformation plugins."""

    name: str  # plugin identifier

    @abc.abstractmethod
    def apply(self, text: str, rng: random.Random) -> tuple[str, dict]:
        """Apply the transformation to *text* using *rng* for deterministic randomness.

        Returns a tuple of ``(transformed_text, metadata)`` where ``metadata`` is a
        dictionary that will be added to the synthetic sample's provenance.
        """
        raise NotImplementedError
