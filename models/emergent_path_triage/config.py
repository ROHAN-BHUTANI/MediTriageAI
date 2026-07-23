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

    # Configuration-driven ablation study toggles
    ablation_router_enabled: bool = True
    ablation_dces_enabled: bool = True
    ablation_engine_enabled: bool = True
    ablation_ctb1_enabled: bool = True
    ablation_ctb2_enabled: bool = True
    ablation_ctb3_enabled: bool = True
    ablation_ctb4_enabled: bool = True
    ablation_multistep_enabled: bool = True

    # ACES specific configuration
    aces_fusion_mode: str = "A0"  # A0: Static, A1: Attn, A2: Attn+Res, A3: Attn+Res+Proto
    aces_num_heads: int = 4
    
    # AMCO specific configuration
    amco_optimization_strategy: str = "STATIC"  # STATIC, HOMOSCEDASTIC, DWA, GRADNORM

    # DCCF (Dynamic Clinical Confidence Framework) configuration
    dccf_confidence_estimator: str = "IDENTITY"  # IDENTITY, TEMPERATURE, VECTOR, DIRICHLET
    dccf_deferral_threshold: float = 0.85

    # CCSM (Clinical Cognitive State Machine) configuration
    closed_loop_enabled: bool = True
    routing_trace_level: str = "STANDARD"
    routing_trace_record_hidden_states: bool | None = None
    routing_trace_record_logits: bool | None = None
    routing_trace_record_probabilities: bool | None = None
    routing_trace_record_reasoning_vectors: bool | None = None
    routing_trace_record_entropy: bool | None = None

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
            
        valid_aces_modes = {"A0", "A1", "A2", "A3"}
        if self.aces_fusion_mode not in valid_aces_modes:
            raise ConfigurationError(
                f"aces_fusion_mode must be one of {valid_aces_modes}, got '{self.aces_fusion_mode}'"
            )
        if self.aces_num_heads <= 0:
            raise ConfigurationError(
                f"aces_num_heads must be positive, got {self.aces_num_heads}"
            )
            
        valid_amco_strategies = {"STATIC", "HOMOSCEDASTIC", "DWA", "GRADNORM"}
        if self.amco_optimization_strategy not in valid_amco_strategies:
            raise ConfigurationError(
                f"amco_optimization_strategy must be one of {valid_amco_strategies}, got '{self.amco_optimization_strategy}'"
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

        # Validate CCSM parameters
        valid_trace_levels = {"MINIMAL", "STANDARD", "FULL"}
        if self.routing_trace_level not in valid_trace_levels:
            raise ConfigurationError(
                f"routing_trace_level must be one of {valid_trace_levels}, got '{self.routing_trace_level}'"
            )

        valid_dccf_strategies = {"IDENTITY", "TEMPERATURE", "VECTOR", "DIRICHLET"}
        if self.dccf_confidence_estimator not in valid_dccf_strategies:
            raise ConfigurationError(
                f"dccf_confidence_estimator must be one of {valid_dccf_strategies}, got '{self.dccf_confidence_estimator}'"
            )
        if not (0.0 <= self.dccf_deferral_threshold <= 1.0):
            raise ConfigurationError(
                f"dccf_deferral_threshold must be between 0.0 and 1.0, got {self.dccf_deferral_threshold}"
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
            "ablation_router_enabled",
            "ablation_dces_enabled",
            "ablation_engine_enabled",
            "ablation_ctb1_enabled",
            "ablation_ctb2_enabled",
            "ablation_ctb3_enabled",
            "ablation_ctb4_enabled",
            "ablation_multistep_enabled",
            "aces_fusion_mode",
            "aces_num_heads",
            "closed_loop_enabled",
            "routing_trace_level",
            "routing_trace_record_hidden_states",
            "routing_trace_record_logits",
            "routing_trace_record_probabilities",
            "routing_trace_record_reasoning_vectors",
            "routing_trace_record_entropy",
            "amco_optimization_strategy",
            "dccf_confidence_estimator",
            "dccf_deferral_threshold",
        }
        
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        try:
            return cls(**filtered)
        except Exception as e:
            raise ConfigurationError(f"Failed to parse configuration: {e}") from e
