"""Data loading, batch inference, and Parquet caching for the MediTriageAI analysis framework."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from models.distilbert_multi import DistilBertMultilingualModel
from models.indic_bert import IndicBertModel
from models.mbert import MBertModel
from models.xlm_roberta import XLMRobertaLargeModel
from src.model import SEVERITY_LABELS, SPECIALIST_CLASSES

# Map config names to base model wrapper classes
MODEL_MAP: dict[str, type] = {
    "xlm_roberta_large": XLMRobertaLargeModel,
    "mbert": MBertModel,
    "distilbert_multilingual": DistilBertMultilingualModel,
    "indic_bert": IndicBertModel,
}


def load_test_dataframe(dataset_csv: Path) -> pd.DataFrame:
    """Load the test split from the dataset CSV."""
    if not dataset_csv.exists():
        raise FileNotFoundError(f"Dataset not found at: {dataset_csv}")
    df = pd.read_csv(dataset_csv)
    df_test = df[df["split"] == "test"].copy()
    # Clean null texts
    df_test = df_test.dropna(subset=["text"])
    return df_test


def _load_checkpoint(model: torch.nn.Module, checkpoint: Path) -> None:
    """Load parameter weights from a checkpoint file into a built model."""
    if not checkpoint.exists():
        raise FileNotFoundError(f"Model checkpoint not found at: {checkpoint}")
    state_dict = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)


def generate_and_cache_predictions(
    model_name: str, config: Any, logger: logging.Logger
) -> pd.DataFrame:
    """Load model checkpoint, run batch inference on the test split, and cache predictions.

    If the Parquet prediction cache file already exists, it is loaded directly.
    """
    cache_path = config.get_prediction_cache_path(model_name)
    if cache_path.exists():
        logger.info(
            f"Found predictions cache for {model_name} at: {cache_path}. Loading..."
        )
        return pd.read_parquet(cache_path)

    logger.info(
        f"Predictions cache NOT found for {model_name}. Running model inference..."
    )

    # Load dataset test split
    df_test = load_test_dataframe(config.dataset_csv)
    logger.info(f"Loaded test split containing {len(df_test)} samples.")

    # Initialize model
    if model_name not in MODEL_MAP:
        raise ValueError(
            f"Unknown model name: {model_name}. Available: {list(MODEL_MAP.keys())}"
        )

    model_cls = MODEL_MAP[model_name]
    model_instance = model_cls()
    tokenizer = model_instance.get_tokenizer()
    built_model = model_instance.build(None)

    # Apply Hinglish phonetic vocab injection if needed
    if model_cls.needs_vocab_injection():
        logger.info(f"Injecting Hinglish vocabulary for model {model_name}...")
        model_instance.inject_vocab(built_model, tokenizer)

    # Load trained weights
    checkpoint_path = config.get_checkpoint_path(model_name)
    logger.info(f"Loading checkpoint weights from: {checkpoint_path}")
    _load_checkpoint(built_model, checkpoint_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Moving model to device: {device}")
    built_model.to(device)
    built_model.eval()

    # Inference in batches to conserve memory
    batch_size = 32
    texts = df_test["text"].tolist()

    specialist_logits_list = []
    severity_logits_list = []

    logger.info(
        f"Running forward passes on {len(texts)} samples (batch_size={batch_size})..."
    )
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]

        # Max length should align with training (max_length=64)
        inputs = tokenizer(
            batch_texts,
            truncation=True,
            padding="max_length",
            max_length=64,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        with torch.no_grad():
            spec_logits, sev_logits = built_model(input_ids, attention_mask)

        specialist_logits_list.append(spec_logits.cpu())
        severity_logits_list.append(sev_logits.cpu())

    # Concatenate and compute probabilities
    all_spec_logits = torch.cat(specialist_logits_list, dim=0).numpy()
    all_sev_logits = torch.cat(severity_logits_list, dim=0).numpy()

    # Compute softmax probabilities
    # Logits dimensions: [N, C_spec], [N, C_sev]
    spec_exp = np.exp(all_spec_logits - np.max(all_spec_logits, axis=-1, keepdims=True))
    all_spec_probs = spec_exp / np.sum(spec_exp, axis=-1, keepdims=True)

    sev_exp = np.exp(all_sev_logits - np.max(all_sev_logits, axis=-1, keepdims=True))
    all_sev_probs = sev_exp / np.sum(sev_exp, axis=-1, keepdims=True)

    # Get class predictions
    pred_spec_ids = np.argmax(all_spec_probs, axis=-1)
    pred_sev_ids = np.argmax(all_sev_probs, axis=-1)

    # Set severity column selector based on presence in CSV
    severity_col = (
        "severity_heuristic"
        if "severity_heuristic" in df_test.columns
        else "severity_label"
    )

    # Instantiate language detector
    from analysis.language_detector import HeuristicLanguageDetector

    lang_detector = HeuristicLanguageDetector()

    out_rows = []
    for idx, (_, row) in enumerate(df_test.iterrows()):
        text = str(row["text"])

        # Ground truth labels (strings)
        true_spec = str(row["department_code"])
        true_sev = str(row[severity_col])

        # Predicted labels (strings)
        pred_spec = SPECIALIST_CLASSES[pred_spec_ids[idx]]
        pred_sev = SEVERITY_LABELS[pred_sev_ids[idx]]

        # Token length using whitespace split
        token_count = len(text.split())

        # Detected language
        detected_lang = lang_detector.detect(text)

        out_rows.append(
            {
                "sample_id": str(row.get("tracking_id", f"idx_{idx}")),
                "text": text,
                "true_specialist": true_spec,
                "pred_specialist": pred_spec,
                "true_severity": true_sev,
                "pred_severity": pred_sev,
                "specialist_logits": all_spec_logits[idx].tolist(),
                "severity_logits": all_sev_logits[idx].tolist(),
                "specialist_probabilities": all_spec_probs[idx].tolist(),
                "severity_probabilities": all_sev_probs[idx].tolist(),
                "language": detected_lang,
                "token_count": token_count,
                "model_name": model_name,
            }
        )

    df_out = pd.DataFrame(out_rows)

    # Save to Parquet format
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(cache_path, index=False)
    logger.info(f"Successfully cached predictions to: {cache_path}")

    return df_out
