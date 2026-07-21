"""Configuration management and schema validation for E-PATH-CO-REASON."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

from models.emergent_path_triage.constants import (
    DEFAULT_LATENT_DIM,
    DEFAULT_ROUTING_DEPTH,
    DEFAULT_NUM_CTBS,
    DEFAULT_TEMPERATURE,
    DEFAULT_ALPHA_SPECIALIST,
    DEFAULT_BETA_SEVERITY,
    DEFAULT_ORTHO_LAMBDA,
    DEFAULT_CONS_LAMBDA,
    DEFAULT_DIV_LAMBDA,
)
from models.emergent_path_triage.exceptions import ConfigurationError


@dataclass
class EmergentPathTriageConfig:
    """Configuration options for E-PATH-CO-REASON.
    
    Exposes schema versioning and performs self-validation during construction.
    """
    num_thought_blocks: int = DEFAULT_NUM_CTBS
    max_path_depth: int = DEFAULT_ROUTING_DEPTH
    latent_dim: int = DEFAULT_LATENT_DIM
    temperature: float = DEFAULT_TEMPERATURE
    
    alpha_specialist: float = DEFAULT_ALPHA_SPECIALIST
    beta_severity: float = DEFAULT_BETA_SEVERITY
    
    ortho_lambda: float = DEFAULT_ORTHO_LAMBDA
    cons_lambda: float = DEFAULT_CONS_LAMBDA
    div_lambda: float = DEFAULT_DIV_LAMBDA

    # DCES specific configuration
    dces_dropout: float = 0.1
    dces_activation: str = "gelu"
    dces_normalization: str = "layernorm"

    # DCRR specific configuration
    routing_hidden_dim: int = 64

    # CTB specific configuration
    ctb_hidden_dim: int = 128
    ctb_dropout: float = 0.1
    ctb_activation: str = "gelu"
    ctb_normalization: str = "layernorm"

    # Prediction Head specific configuration
    head_hidden_dim: int = 64
    head_dropout: float = 0.1
    head_activation: str = "gelu"

    # Versioning metadata
    schema_version: str = "1.0"
    architecture_version: str = "1.0.0"
    compatibility_version: str = "1.0"

    def __post_init__(self) -> None:
        """Automatically validate configuration settings on instantiation."""
        self.validate()

    def validate(self) -> None:
        """Ensure all configuration bounds are clinically and logically valid."""
        if self.num_thought_blocks <= 0:
            raise ConfigurationError(
                f"num_thought_blocks must be positive, got {self.num_thought_blocks}"
            )
        if self.max_path_depth <= 0:
            raise ConfigurationError(
                f"max_path_depth must be positive, got {self.max_path_depth}"
            )
        if self.latent_dim <= 0 or self.latent_dim % 2 != 0:
            raise ConfigurationError(
                f"latent_dim must be a positive multiple of 2, got {self.latent_dim}"
            )
        if self.temperature <= 0:
            raise ConfigurationError(
                f"temperature must be strictly positive, got {self.temperature}"
            )

        # Loss weights coefficients must be non-negative
        for name, value in {
            "alpha_specialist": self.alpha_specialist,
            "beta_severity": self.beta_severity,
            "ortho_lambda": self.ortho_lambda,
            "cons_lambda": self.cons_lambda,
            "div_lambda": self.div_lambda,
        }.items():
            if value < 0:
                raise ConfigurationError(
                    f"Loss coefficient {name} must be non-negative, got {value}"
                )

        # Validate DCES parameters
        if not (0.0 <= self.dces_dropout <= 1.0):
            raise ConfigurationError(
                f"dces_dropout must be in [0.0, 1.0], got {self.dces_dropout}"
            )
        valid_activations = {"gelu", "relu", "silu", "tanh"}
        if self.dces_activation not in valid_activations:
            raise ConfigurationError(
                f"dces_activation must be one of {valid_activations}, got '{self.dces_activation}'"
            )
        valid_norms = {"layernorm", "none"}
        if self.dces_normalization not in valid_norms:
            raise ConfigurationError(
                f"dces_normalization must be one of {valid_norms}, got '{self.dces_normalization}'"
            )

        # Validate DCRR parameters
        if self.routing_hidden_dim <= 0:
            raise ConfigurationError(
                f"routing_hidden_dim must be positive, got {self.routing_hidden_dim}"
            )

        # Validate CTB parameters
        if self.ctb_hidden_dim <= 0:
            raise ConfigurationError(
                f"ctb_hidden_dim must be positive, got {self.ctb_hidden_dim}"
            )
        if not (0.0 <= self.ctb_dropout <= 1.0):
            raise ConfigurationError(
                f"ctb_dropout must be in [0.0, 1.0], got {self.ctb_dropout}"
            )
        if self.ctb_activation not in valid_activations:
            raise ConfigurationError(
                f"ctb_activation must be one of {valid_activations}, got '{self.ctb_activation}'"
            )
        if self.ctb_normalization not in valid_norms:
            raise ConfigurationError(
                f"ctb_normalization must be one of {valid_norms}, got '{self.ctb_normalization}'"
            )

        # Validate Head parameters
        if self.head_hidden_dim <= 0:
            raise ConfigurationError(
                f"head_hidden_dim must be positive, got {self.head_hidden_dim}"
            )
        if not (0.0 <= self.head_dropout <= 1.0):
            raise ConfigurationError(
                f"head_dropout must be in [0.0, 1.0], got {self.head_dropout}"
            )
        if self.head_activation not in valid_activations:
            raise ConfigurationError(
                f"head_activation must be one of {valid_activations}, got '{self.head_activation}'"
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to a deterministic dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> EmergentPathTriageConfig:
        """Parse, validate, and construct configuration from a dictionary."""
        if not data:
            return cls()
        
        # Check compatibility version if present in input dictionary
        if "compatibility_version" in data:
            input_ver = str(data["compatibility_version"])
            current_ver = "1.0"
            if input_ver != current_ver:
                raise ConfigurationError(
                    f"Incompatible configuration version: input schema {input_ver} "
                    f"is not compatible with current schema {current_ver}."
                )

        valid_keys = {
            "num_thought_blocks",
            "max_path_depth",
            "latent_dim",
            "temperature",
            "alpha_specialist",
            "beta_severity",
            "ortho_lambda",
            "cons_lambda",
            "div_lambda",
            "dces_dropout",
            "dces_activation",
            "dces_normalization",
            "routing_hidden_dim",
            "ctb_hidden_dim",
            "ctb_dropout",
            "ctb_activation",
            "ctb_normalization",
            "head_hidden_dim",
            "head_dropout",
            "head_activation",
            "schema_version",
            "architecture_version",
            "compatibility_version",
        }
        
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        try:
            return cls(**filtered)
        except Exception as e:
            raise ConfigurationError(f"Failed to parse configuration: {e}") from e
