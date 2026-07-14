"""Shared model layers for MediTriageAI."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import XLMRobertaModel

SPECIALIST_CLASSES = [
    "CARDIO_PULM", "ED", "ENT_OPHTHALMO", "GEN_MED", "GI", "NEURO",
    "OBGYN", "ONCOLOGY_HEME", "ORTHO", "PEDS", "PSYCH", "RENAL_URO", "SURGERY"
]
SEVERITY_LABELS = ["S1", "S2", "S3", "S4", "S5"]
DEFAULT_ALPHA = 1.0
DEFAULT_BETA = 1.2


class JointLossWeights:
    def __init__(self, alpha_specialist: float = DEFAULT_ALPHA, beta_severity: float = DEFAULT_BETA):
        self.alpha_specialist = alpha_specialist
        self.beta_severity = beta_severity


class MediTriageTransformer(nn.Module):
    def __init__(self, encoder: XLMRobertaModel, hidden_dropout_prob: float = 0.1) -> None:
        super().__init__()
        self.encoder = encoder
        hidden_size = encoder.config.hidden_size
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.classifier_specialist = nn.Linear(hidden_size, len(SPECIALIST_CLASSES))
        self.classifier_severity = nn.Linear(hidden_size, len(SEVERITY_LABELS))

    def get_input_embeddings(self):
        return self.encoder.get_input_embeddings()

    def resize_token_embeddings(self, new_num_tokens: int) -> torch.nn.Embedding | None:
        return self.encoder.resize_token_embeddings(new_num_tokens)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoder_output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_representation = self.dropout(encoder_output.last_hidden_state[:, 0, :])
        return self.classifier_specialist(cls_representation), self.classifier_severity(cls_representation)


class JointLoss(nn.Module):
    def __init__(self, weight: JointLossWeights = None):
        super().__init__()
        self.weight = weight or JointLossWeights()
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, specialist_logits: torch.Tensor, severity_logits: torch.Tensor, labels_specialist: torch.Tensor, labels_severity: torch.Tensor) -> dict[str, torch.Tensor | None]:
        specialist_loss = self.cross_entropy(specialist_logits, labels_specialist)
        severity_loss = self.cross_entropy(severity_logits, labels_severity)
        joint_loss = self.weight.alpha_specialist * specialist_loss + self.weight.beta_severity * severity_loss
        return {"joint_loss": joint_loss, "specialist_loss": specialist_loss.detach(), "severity_loss": severity_loss.detach()}
