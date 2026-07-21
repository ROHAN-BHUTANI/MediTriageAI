"""Centralized architectural constants for E-PATH-CO-REASON."""

from __future__ import annotations

# Class specifications
NUM_SPECIALIST_CLASSES: int = 13
NUM_SEVERITY_LABELS: int = 5

# Default hyper-parameters
DEFAULT_LATENT_DIM: int = 64
DEFAULT_ROUTING_DEPTH: int = 3
DEFAULT_NUM_CTBS: int = 4
DEFAULT_TEMPERATURE: float = 1.0

# Loss balancing constants
DEFAULT_ALPHA_SPECIALIST: float = 1.0
DEFAULT_BETA_SEVERITY: float = 1.2
DEFAULT_ORTHO_LAMBDA: float = 0.1
DEFAULT_CONS_LAMBDA: float = 0.5
DEFAULT_DIV_LAMBDA: float = 0.1
