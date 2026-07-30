"""Model zoo for MediTriageAI."""

from .base_model import BaseMediTriageModel
from .distilbert_multi import DistilBertMultilingualModel
from .emergent_path_triage import EmergentPathTriageModel
from .indic_bert import IndicBertModel
from .mbert import MBertModel
from .xlm_roberta import XLMRobertaLargeModel

MODEL_REGISTRY = {
    "1": XLMRobertaLargeModel,
    "2": MBertModel,
    "3": DistilBertMultilingualModel,
    "4": IndicBertModel,
    "5": EmergentPathTriageModel,
}

__all__ = [
    "MODEL_REGISTRY",
    "BaseMediTriageModel",
    "DistilBertMultilingualModel",
    "EmergentPathTriageModel",
    "IndicBertModel",
    "MBertModel",
    "XLMRobertaLargeModel",
]
