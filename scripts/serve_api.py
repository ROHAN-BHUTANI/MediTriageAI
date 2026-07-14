"""FastAPI inference service for MediTriageAI."""

from __future__ import annotations

import secrets
import sys
from pathlib import Path
from typing import Dict, Any, List

import torch
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.mbert import MBertModel

# Constants
SPECIALIST_LABELS = [
    "CARDIO_PULM",
    "ED",
    "ENT_OPHTHALMO",
    "GEN_MED",
    "GI",
    "NEURO",
    "OBGYN",
    "ONCOLOGY_HEME",
    "ORTHO",
    "PEDS",
    "PSYCH",
    "RENAL_URO",
    "SURGERY",
]

SEVERITY_LABELS = ["S1 URGENT", "S2 EMERGENT", "S3 URGENT", "S4", "S5"]

# Load model and tokenizer
model_meta = MBertModel()
tokenizer = model_meta.get_tokenizer()
built_model = model_meta.build(None)
if MBertModel.needs_vocab_injection():
    model_meta.inject_vocab(built_model, tokenizer)

checkpoint_path = REPO_ROOT / "results" / "mbert" / "checkpoint.pt"
if checkpoint_path.exists():
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    built_model.load_state_dict(state_dict, strict=False)
    print(f"Loaded mBERT checkpoint from {checkpoint_path}")
else:
    print(f"WARNING: Checkpoint not found at {checkpoint_path}. Running with random initialization.")

built_model.eval()

# Initialize FastAPI app
app = FastAPI(
    title="MediTriageAI Clinical Triage API",
    description="FastAPI service exposing dual-head model inference for specialist routing and ESI severity triage.",
    version="1.0.0",
)

import os

API_USER = os.environ.get("MEDITRIAGE_API_USER")
API_PASS = os.environ.get("MEDITRIAGE_API_PASS")

if not API_USER or not API_PASS:
    raise RuntimeError(
        "API credentials are not set. Please set MEDITRIAGE_API_USER and MEDITRIAGE_API_PASS environment variables."
    )

security = HTTPBasic()

def authenticate(credentials: HTTPBasicCredentials = Depends(security)) -> str:
    correct_username = secrets.compare_digest(credentials.username, API_USER)
    correct_password = secrets.compare_digest(credentials.password, API_PASS)
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


class PredictionRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=5,
        description="Patient clinical transcription or symptom description text.",
        examples=["Patient has severe abdominal pain and fever."],
    )

class SpecialistPrediction(BaseModel):
    label: str
    confidence: float

class PredictionResponse(BaseModel):
    input_text: str
    specialist_routing: List[SpecialistPrediction]
    severity_triage: Dict[str, Any]
    red_flags_matched: List[str]
    requires_manual_review: bool
    disclaimer: str

RED_FLAG_KEYWORDS = [
    "chest pain",
    "radiation",
    "loss of consciousness",
    "severe bleeding",
    "stroke",
    "slurred speech",
    "suicide",
    "gunshot"
]

LOW_CONFIDENCE_THRESHOLD = 0.60



@app.post("/predict", response_model=PredictionResponse, summary="Perform Clinical Triage Prediction")
def predict(request: PredictionRequest, username: str = Depends(authenticate)) -> Dict[str, Any]:
    """Exposes dual-head model prediction for specialist routing (13 classes) and severity triage (5 ESI classes)."""
    text_lower = request.text.lower()
    red_flags_matched = [kw for kw in RED_FLAG_KEYWORDS if kw in text_lower]
    
    inputs = tokenizer(request.text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    
    with torch.no_grad():
        specialist_logits, severity_logits = built_model(inputs["input_ids"], inputs["attention_mask"])
        
    specialist_probs = torch.softmax(specialist_logits[0], dim=-1)
    severity_probs = torch.softmax(severity_logits[0], dim=-1)
    
    # Top 3 Specialist predictions
    topk_vals, topk_idxs = torch.topk(specialist_probs, k=3, dim=-1)
    specialist_routing = []
    for val, idx in zip(topk_vals, topk_idxs):
        specialist_routing.append({
            "label": SPECIALIST_LABELS[idx.item()],
            "confidence": float(val.item())
        })
        
    sev_val, sev_idx = torch.max(severity_probs, dim=-1)
    sev_label = SEVERITY_LABELS[sev_idx.item()]
    sev_conf = float(sev_val.item())
    
    # Escalation Logic
    highest_spec_conf = specialist_routing[0]["confidence"]
    requires_manual_review = len(red_flags_matched) > 0 or highest_spec_conf < LOW_CONFIDENCE_THRESHOLD
    
    return {
        "input_text": request.text,
        "specialist_routing": specialist_routing,
        "severity_triage": {
            "label": sev_label,
            "confidence": sev_conf,
        },
        "red_flags_matched": red_flags_matched,
        "requires_manual_review": requires_manual_review,
        "disclaimer": "RESEARCH PROTOTYPE - NOT clinically validated. Do NOT use for real triage decisions.",
    }


@app.get("/health", summary="Health Check")
def health() -> Dict[str, str]:
    return {"status": "healthy"}
