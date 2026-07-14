"""Model zoo for MediTriageAI."""

from .base_model import BaseMediTriageModel
from .distilbert_multi import DistilBertMultilingualModel
from .indic_bert import IndicBertModel
from .mbert import MBertModel
from .xlm_roberta import XLMRobertaLargeModel

MODEL_REGISTRY = {
    "1": XLMRobertaLargeModel,
    "2": MBertModel,
    "3": DistilBertMultilingualModel,
    "4": IndicBertModel,
}

__all__ = [
    "BaseMediTriageModel",
    "XLMRobertaLargeModel",
    "MBertModel",
    "DistilBertMultilingualModel",
    "IndicBertModel",
    "MODEL_REGISTRY",
]
