"""Clinical Linguistic Variation Subsystem for MediTriageAI Multilingual Pipeline."""

from meditriage.multilingual.variation.config import VariationConfig
from meditriage.multilingual.variation.engine import ClinicalLinguisticVariationEngine
from meditriage.multilingual.variation.generators import (
    BaseVariationGenerator,
    get_all_generators,
    get_generator_by_name,
)
from meditriage.multilingual.variation.validator import (
    SemanticVariationValidator,
    VariationValidationResult,
)

__all__ = [
    "VariationConfig",
    "ClinicalLinguisticVariationEngine",
    "BaseVariationGenerator",
    "get_all_generators",
    "get_generator_by_name",
    "SemanticVariationValidator",
    "VariationValidationResult",
]
