"""Stage 4 – Multi-Factor Diversity Scoring.

Computes a composite diversity score for every sample based on:
  1. Lexical diversity     – Type-Token Ratio
  2. Semantic diversity    – TF-IDF distance from cluster centroid
  3. Symptom diversity     – Count of medical/symptom n-grams
  4. Language diversity    – Bonus for non-majority language
  5. Text-length diversity – Deviation from department mean length

Writes:
  stage4_diversity_scores.parquet – full dataset with diversity columns
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from reconstruction.config import ReconstructionConfig

logger = logging.getLogger(__name__)

STAGE_NAME = "stage4_diversity"

_TOKEN_RE = re.compile(r"\w+")

# Common medical/symptom terms for symptom diversity scoring
_SYMPTOM_TERMS = frozenset([
    "pain", "ache", "swelling", "bleeding", "fever", "cough", "vomit",
    "nausea", "dizziness", "headache", "fracture", "injury", "wound",
    "burn", "rash", "itching", "numbness", "weakness", "fatigue",
    "breathless", "shortness", "breath", "chest", "abdominal", "stomach",
    "diarrhea", "constipation", "infection", "inflammation", "trauma",
    "laceration", "contusion", "sprain", "strain", "dislocation",
    "concussion", "seizure", "unconscious", "syncope", "palpitation",
    "hypertension", "diabetes", "asthma", "allergy", "anxiety",
    "depression", "insomnia", "tremor", "paralysis", "vision", "hearing",
    # Hindi/Hinglish symptom terms
    "dard", "sujan", "bukhar", "khoon", "khansi", "ulti", "chakkar",
    "sar", "toot", "chot", "jalna", "khujli", "kamzori", "thakan",
    "saans", "seena", "pet", "dast", "infection", "soojan",
])


def compute_lexical_diversity(text: str) -> float:
    """Type-Token Ratio of the text.

    Args:
        text: Input text.

    Returns:
        Float in [0, 1].
    """
    tokens = _TOKEN_RE.findall(text.lower())
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def compute_symptom_diversity(text: str) -> float:
    """Fraction of known symptom terms present in the text.

    Args:
        text: Input text.

    Returns:
        Float in [0, 1].
    """
    tokens = set(_TOKEN_RE.findall(text.lower()))
    if not tokens:
        return 0.0
    symptom_count = len(tokens & _SYMPTOM_TERMS)
    return min(symptom_count / 5.0, 1.0)  # Normalize: 5+ symptoms = max


def compute_language_diversity(language: str, majority_language: str) -> float:
    """Score based on whether the sample is in a non-majority language.

    Args:
        language: Sample language tag.
        majority_language: Most common language in the dataset.

    Returns:
        Float: 1.0 for non-majority, 0.0 for majority.
    """
    if pd.isna(language) or pd.isna(majority_language):
        return 0.0
    return 0.0 if language == majority_language else 1.0


def compute_text_length_diversity(length: int, mean_length: float, std_length: float) -> float:
    """Score based on deviation from department mean text length.

    Texts that are unusually short or long score higher (more diverse).

    Args:
        length: Character length of the text.
        mean_length: Department mean length.
        std_length: Department std length.

    Returns:
        Float in [0, 1].
    """
    if std_length < 1.0:
        return 0.0
    z = abs(length - mean_length) / std_length
    return min(z / 3.0, 1.0)  # z-score of 3+ = max diversity


def score_department(
    dept_df: pd.DataFrame,
    cfg: ReconstructionConfig,
    majority_language: str,
) -> pd.DataFrame:
    """Compute all diversity scores for a single department.

    Args:
        dept_df: DataFrame slice for one department (must have cluster_id).
        cfg: Reconstruction configuration.
        majority_language: Most common language in the full dataset.

    Returns:
        DataFrame with added diversity score columns.
    """
    dept_df = dept_df.copy()
    texts = dept_df["raw_text"].tolist()
    n = len(texts)

    # 1. Lexical diversity
    dept_df["div_lexical"] = dept_df["raw_text"].apply(compute_lexical_diversity)

    # 2. Semantic diversity – distance from cluster centroid in TF-IDF space
    semantic_scores = np.zeros(n, dtype=np.float64)
    if n > 1:
        vectorizer = TfidfVectorizer(
            max_features=cfg.tfidf_max_features,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        try:
            tfidf = vectorizer.fit_transform(texts)
            for cluster_id in dept_df["cluster_id"].unique():
                mask = (dept_df["cluster_id"].values == cluster_id)
                cluster_vecs = tfidf[mask]
                if cluster_vecs.shape[0] <= 1:
                    continue
                centroid = np.asarray(cluster_vecs.mean(axis=0))
                # Cosine distance from centroid
                from sklearn.metrics.pairwise import cosine_distances
                dists = cosine_distances(cluster_vecs, centroid).ravel()
                # Normalize to [0, 1]
                max_dist = dists.max() if dists.max() > 0 else 1.0
                semantic_scores[mask] = dists / max_dist
        except ValueError:
            pass

    dept_df["div_semantic"] = semantic_scores

    # 3. Symptom diversity
    dept_df["div_symptom"] = dept_df["raw_text"].apply(compute_symptom_diversity)

    # 4. Language diversity
    dept_df["div_language"] = dept_df["language"].apply(
        lambda lang: compute_language_diversity(lang, majority_language)
    )

    # 5. Text-length diversity
    lengths = dept_df["raw_text"].str.len()
    mean_len = float(lengths.mean())
    std_len = float(lengths.std()) if len(lengths) > 1 else 1.0
    dept_df["div_text_length"] = lengths.apply(
        lambda l: compute_text_length_diversity(l, mean_len, std_len)
    )

    # Composite score
    w = cfg.diversity_weights
    dept_df["diversity_score"] = (
        w["lexical"] * dept_df["div_lexical"]
        + w["semantic"] * dept_df["div_semantic"]
        + w["symptom"] * dept_df["div_symptom"]
        + w["language"] * dept_df["div_language"]
        + w["text_length"] * dept_df["div_text_length"]
    )

    return dept_df


def run(df: pd.DataFrame, cfg: ReconstructionConfig) -> pd.DataFrame:
    """Execute Stage 4: compute diversity scores and write artifact.

    Args:
        df: Clustered DataFrame from Stage 3 (must have cluster_id).
        cfg: Reconstruction configuration.

    Returns:
        DataFrame with diversity score columns added.
    """
    out_dir = Path(cfg.output_directory)
    out_dir.mkdir(parents=True, exist_ok=True)

    scores_path = out_dir / "stage4_diversity_scores.parquet"

    # Resume support
    if scores_path.exists():
        logger.info("Stage 4 artifacts found, resuming from %s", scores_path)
        return pd.read_parquet(scores_path)

    # Determine majority language
    majority_language = df["language"].mode().iloc[0] if "language" in df.columns else "en"

    scored_dfs = []
    for dept in df["department"].unique():
        dept_df = df[df["department"] == dept]
        logger.info("  Scoring %s: %d samples", dept, len(dept_df))
        scored = score_department(dept_df, cfg, majority_language)
        scored_dfs.append(scored)

    result = pd.concat(scored_dfs, ignore_index=True)

    result.to_parquet(scores_path, index=False)
    logger.info("Stage 4 complete. Scores written to %s", scores_path)
    return result
