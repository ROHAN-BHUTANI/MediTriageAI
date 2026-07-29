"""Custom exception hierarchy for E-PATH-CO-REASON."""

from __future__ import annotations


class MediTriageError(Exception):
    """Base exception for all E-PATH-CO-REASON software components."""


class ConfigurationError(MediTriageError):
    """Raised when configuration parameters or version validation fails."""


class RoutingError(MediTriageError):
    """Raised when routing decisions, state paths, or entropy checks fail."""


class InterfaceError(MediTriageError):
    """Raised when interface contracts (shape, dtype, or device expectations) are violated."""


class CompatibilityError(MediTriageError):
    """Raised when checkpoint loading version verification fails."""
