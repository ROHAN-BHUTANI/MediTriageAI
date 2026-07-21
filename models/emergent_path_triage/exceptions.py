"""Custom exception hierarchy for E-PATH-CO-REASON."""

from __future__ import annotations


class MediTriageError(Exception):
    """Base exception for all E-PATH-CO-REASON software components."""
    pass


class ConfigurationError(MediTriageError):
    """Raised when configuration parameters or version validation fails."""
    pass


class RoutingError(MediTriageError):
    """Raised when routing decisions, state paths, or entropy checks fail."""
    pass


class InterfaceError(MediTriageError):
    """Raised when interface contracts (shape, dtype, or device expectations) are violated."""
    pass


class CompatibilityError(MediTriageError):
    """Raised when checkpoint loading version verification fails."""
    pass
