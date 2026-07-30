"""Configuration settings for the MediTriageAI analysis framework."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

# Base directory of the repository
REPO_ROOT = Path(__file__).resolve().parent.parent

# Dataset paths
DEFAULT_DATASET_CSV = REPO_ROOT / "meditriage" / "data" / "processed" / "dataset.csv"

# Model metadata
MODELS_TO_ANALYZE = (
    "xlm_roberta_large",
    "mbert",
    "distilbert_multilingual",
    "indic_bert",
)

# Output paths
ANALYSIS_ROOT = REPO_ROOT / "analysis"
CACHE_DIR = ANALYSIS_ROOT / "cache"
PREDICTIONS_CACHE_DIR = CACHE_DIR / "predictions"
EMBEDDINGS_CACHE_DIR = CACHE_DIR / "embeddings"
METADATA_CACHE_DIR = CACHE_DIR / "metadata"

RESULTS_ROOT = ANALYSIS_ROOT / "results"


class AnalysisConfig(NamedTuple):
    dataset_csv: Path = DEFAULT_DATASET_CSV
    models: tuple[str, ...] = MODELS_TO_ANALYZE

    # Seeding
    random_seed: int = 42

    # Bootstrap
    bootstrap_iterations: int = 1000

    # Visualizations
    plot_dpi: int = 300

    # Confidence threshold boundaries for error taxonomy
    high_confidence_threshold: float = 0.70
    low_confidence_specialist_threshold: float = 0.35
    low_confidence_severity_threshold: float = 0.40

    # Directories
    results_root: Path = RESULTS_ROOT
    predictions_cache_dir: Path = PREDICTIONS_CACHE_DIR
    embeddings_cache_dir: Path = EMBEDDINGS_CACHE_DIR
    metadata_cache_dir: Path = METADATA_CACHE_DIR

    # Model checkpoints mapping
    def get_checkpoint_path(self, model_name: str) -> Path:
        return REPO_ROOT / "results" / model_name / "checkpoint.pt"

    # Prediction Parquet cache path
    def get_prediction_cache_path(self, model_name: str) -> Path:
        return self.predictions_cache_dir / f"{model_name}.parquet"


# Global configuration instance
config = AnalysisConfig()
