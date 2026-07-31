"""Unit tests for the Dataset Reconstruction Engine – Stages 1-5."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from reconstruction.backends.factory import create_backend
from reconstruction.backends.tfidf import TfidfBackend
from reconstruction.config import ReconstructionConfig
from reconstruction.stage1_load import generate_profile, load_dataset
from reconstruction.stage2_clean import clean_dataset, is_valid_text, normalize_text
from reconstruction.stage3_cluster import _determine_n_clusters, cluster_department
from reconstruction.stage4_diversity import (
    compute_language_diversity,
    compute_lexical_diversity,
    compute_symptom_diversity,
    compute_text_length_diversity,
)
from reconstruction.stage5_undersample import (
    select_from_cluster,
    undersample_department,
)

# ─── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create a small synthetic dataset for testing."""
    rows = []
    departments = ["ORTHO", "NEURO", "PEDS"]
    for i, dept in enumerate(departments):
        for j in range(100):
            rows.append(
                {
                    "id": f"sample_{i}_{j}",
                    "split": "train",
                    "dataset_source": "test",
                    "language": "en" if j % 5 != 0 else "hi-en",
                    "raw_text": f"Patient {j} has {dept.lower()} symptoms including pain and swelling in area {j}",
                    "department": dept,
                    "triage_level": f"S{(j % 5) + 1}",
                }
            )
    # Add some invalid rows
    rows.append(
        {
            "id": "bad1",
            "split": "train",
            "dataset_source": "test",
            "language": "en",
            "raw_text": None,
            "department": "ORTHO",
            "triage_level": "S1",
        }
    )
    rows.append(
        {
            "id": "bad2",
            "split": "train",
            "dataset_source": "test",
            "language": "en",
            "raw_text": "ab",
            "department": "ORTHO",
            "triage_level": "S1",
        }
    )
    rows.append(
        {
            "id": "bad3",
            "split": "train",
            "dataset_source": "test",
            "language": "en",
            "raw_text": "12345",
            "department": "ORTHO",
            "triage_level": "S1",
        }
    )
    rows.append(
        {
            "id": "bad4",
            "split": "train",
            "dataset_source": "test",
            "language": "en",
            "raw_text": "Valid text here",
            "department": None,
            "triage_level": "S1",
        }
    )
    return pd.DataFrame(rows)


@pytest.fixture
def cfg(tmp_path: Path) -> ReconstructionConfig:
    """Create a test config pointing to tmp_path."""
    return ReconstructionConfig(
        target_class_size=50,
        output_directory=str(tmp_path / "reconstruction_output"),
        random_seed=42,
        min_text_length=3,
        max_text_length=50000,
    )


# ─── Config Tests ────────────────────────────────────────────────────────


class TestConfig:
    def test_save_and_load(self, tmp_path: Path):
        cfg = ReconstructionConfig(target_class_size=999)
        path = tmp_path / "cfg.json"
        cfg.save(path)
        loaded = ReconstructionConfig.load(path)
        assert loaded.target_class_size == 999

    def test_from_overrides(self):
        cfg = ReconstructionConfig.from_overrides({"target_class_size": 123})
        assert cfg.target_class_size == 123
        assert cfg.random_seed == 42  # default preserved


# ─── Stage 1 Tests ───────────────────────────────────────────────────────


class TestStage1:
    def test_load_parquet(self, sample_df: pd.DataFrame, tmp_path: Path):
        path = tmp_path / "test.parquet"
        sample_df.to_parquet(path, index=False)
        cfg = ReconstructionConfig(dataset_path=str(path))
        df = load_dataset(cfg)
        assert len(df) == len(sample_df)

    def test_load_csv(self, sample_df: pd.DataFrame, tmp_path: Path):
        path = tmp_path / "test.csv"
        sample_df.to_csv(path, index=False)
        cfg = ReconstructionConfig(dataset_path=str(path))
        df = load_dataset(cfg)
        assert len(df) == len(sample_df)

    def test_load_missing_file(self):
        cfg = ReconstructionConfig(dataset_path="nonexistent.parquet")
        with pytest.raises(FileNotFoundError):
            load_dataset(cfg)

    def test_load_missing_columns(self, tmp_path: Path):
        df = pd.DataFrame({"foo": [1, 2, 3]})
        path = tmp_path / "bad.parquet"
        df.to_parquet(path, index=False)
        cfg = ReconstructionConfig(dataset_path=str(path))
        with pytest.raises(ValueError, match="missing required columns"):
            load_dataset(cfg)

    def test_generate_profile(self, sample_df: pd.DataFrame):
        profile = generate_profile(sample_df)
        assert profile["total_rows"] == len(sample_df)
        assert "department_distribution" in profile
        assert "ORTHO" in profile["department_distribution"]


# ─── Stage 2 Tests ───────────────────────────────────────────────────────


class TestStage2:
    def test_normalize_text(self):
        assert normalize_text("  hello   world  ") == "hello world"
        assert normalize_text("test\u00a0space") == "test space"
        assert normalize_text("  ") == ""

    def test_is_valid_text(self):
        assert is_valid_text("Hello world", 3, 50000) is True
        assert is_valid_text("ab", 3, 50000) is False  # too short
        assert is_valid_text("12345", 3, 50000) is False  # no letters
        assert is_valid_text("", 3, 50000) is False

    def test_clean_drops_invalid(
        self, sample_df: pd.DataFrame, cfg: ReconstructionConfig
    ):
        df_clean, report = clean_dataset(sample_df, cfg)
        # Should have dropped: 1 null text, 1 too short, 1 no letters, 1 null dept
        assert report["dropped"]["missing_raw_text"] == 1
        assert report["dropped"]["missing_department"] == 1
        assert report["dropped"]["invalid_text"] >= 1
        assert len(df_clean) < len(sample_df)


# ─── Backend Tests ───────────────────────────────────────────────────────


class TestClusterBackend:
    def test_tfidf_backend_interface(self):
        backend = TfidfBackend(max_features=100)
        texts = [f"Patient has pain in area {i}" for i in range(50)]
        backend.fit(texts)
        features = backend.encode(texts)
        assert features.shape[0] == 50
        assert features.shape[1] <= 100
        labels = backend.cluster(features, n_clusters=3)
        assert labels.shape == (50,)
        assert len(set(labels)) <= 3

    def test_tfidf_backend_name(self):
        assert TfidfBackend().name == "TF-IDF"

    def test_factory_tfidf(self):
        cfg = ReconstructionConfig(embedding_model="tfidf")
        backend = create_backend(cfg)
        assert isinstance(backend, TfidfBackend)

    def test_factory_unknown(self):
        cfg = ReconstructionConfig(embedding_model="nonexistent")
        with pytest.raises(ValueError, match="Unknown embedding_model"):
            create_backend(cfg)

    def test_encode_before_fit_raises(self):
        backend = TfidfBackend()
        with pytest.raises(RuntimeError):
            backend.encode(["hello"])


# ─── Stage 3 Tests ───────────────────────────────────────────────────────


class TestStage3:
    def test_determine_n_clusters(self):
        assert _determine_n_clusters(10, 50, 5) == 2
        assert _determine_n_clusters(100, 50, 5) == 10
        assert _determine_n_clusters(10000, 50, 5) == 50

    def test_cluster_department_single(self):
        backend = TfidfBackend()
        labels = cluster_department(["one sample"], backend, ReconstructionConfig())
        assert len(labels) == 1
        assert labels[0] == 0

    def test_cluster_department_multiple(self):
        backend = TfidfBackend()
        texts = [f"Patient has pain in area {i}" for i in range(100)]
        labels = cluster_department(texts, backend, ReconstructionConfig())
        assert len(labels) == 100
        assert len(set(labels)) >= 1

    def test_multiple_batches(self):
        backend = TfidfBackend()
        texts = [f"Patient has pain in area {i}" for i in range(100)]
        cfg = ReconstructionConfig(cluster_batch_size=30)
        labels = cluster_department(texts, backend, cfg, dept_name="TEST")
        assert len(labels) == 100

    def test_deterministic_batching(self):
        backend1 = TfidfBackend()
        backend2 = TfidfBackend()
        texts = [f"Patient has symptom {i % 10} and condition {i}" for i in range(100)]
        cfg = ReconstructionConfig(cluster_batch_size=25, random_seed=42)

        labels1 = cluster_department(texts, backend1, cfg)
        labels2 = cluster_department(texts, backend2, cfg)
        assert np.array_equal(labels1, labels2)

    def test_cluster_id_uniqueness(self):
        backend = TfidfBackend()
        texts = [f"Patient has symptom {i}" for i in range(100)]
        cfg = ReconstructionConfig(cluster_batch_size=20)
        labels = cluster_department(texts, backend, cfg, start_cluster_id=10)

        # Unique clusters in batch 1, 2, 3... must be completely disjoint
        # Since cluster_offset increases monotonically, the min label must be >= start_cluster_id
        assert labels.min() >= 10
        # All batch cluster label assignments should be distinct across batch boundaries
        b1 = labels[:20]
        b2 = labels[20:40]
        assert len(set(b1).intersection(set(b2))) == 0

    def test_batch_boundary_correctness(self):
        backend = TfidfBackend()
        # 35 items with batch_size=10 -> 4 batches (10, 10, 10, 5)
        texts = [f"Complaint number {i} with distinct text" for i in range(35)]
        cfg = ReconstructionConfig(cluster_batch_size=10)
        labels = cluster_department(texts, backend, cfg)
        assert len(labels) == 35
        # Verify boundary slicing preserves exact element count
        assert not np.isnan(labels).any()


# ─── Stage 4 Tests ───────────────────────────────────────────────────────


class TestStage4:
    def test_lexical_diversity(self):
        # All unique words -> TTR = 1.0
        assert compute_lexical_diversity("one two three four") == 1.0
        # Repeated words -> TTR < 1.0
        assert compute_lexical_diversity("the the the the") == 0.25
        # Empty
        assert compute_lexical_diversity("") == 0.0

    def test_symptom_diversity(self):
        # Text with known symptoms
        score = compute_symptom_diversity("patient has pain and fever with swelling")
        assert score > 0.0
        # Text with no symptoms
        score = compute_symptom_diversity("hello world foo bar")
        assert score == 0.0

    def test_language_diversity(self):
        assert compute_language_diversity("hi-en", "en") == 1.0
        assert compute_language_diversity("en", "en") == 0.0

    def test_text_length_diversity(self):
        # At the mean -> 0 diversity
        assert compute_text_length_diversity(100, 100.0, 20.0) == 0.0
        # 3 std devs away -> max
        assert compute_text_length_diversity(160, 100.0, 20.0) == 1.0


# ─── Stage 5 Tests ───────────────────────────────────────────────────────


class TestStage5:
    def test_select_from_cluster_under_budget(self):
        df = pd.DataFrame(
            {
                "raw_text": ["a", "b", "c"],
                "diversity_score": [0.5, 0.8, 0.3],
            }
        )
        result = select_from_cluster(df, budget=5)
        assert len(result) == 3  # passthrough

    def test_select_from_cluster_over_budget(self):
        df = pd.DataFrame(
            {
                "raw_text": [f"text_{i}" for i in range(10)],
                "diversity_score": list(range(10)),
            }
        )
        result = select_from_cluster(df, budget=3)
        assert len(result) == 3
        # Should have the 3 highest scores (7, 8, 9)
        assert result["diversity_score"].min() >= 7

    def test_undersample_department(self):
        n = 200
        df = pd.DataFrame(
            {
                "department": ["ORTHO"] * n,
                "cluster_id": [i % 5 for i in range(n)],
                "diversity_score": np.random.rand(n),
                "raw_text": [f"text {i}" for i in range(n)],
                "language": ["en"] * n,
            }
        )
        selected, report = undersample_department(df, target_size=50)
        assert len(selected) == 50
        assert report["action"] == "undersampled"
        # All clusters should be represented
        assert selected["cluster_id"].nunique() == 5

    def test_undersample_passthrough(self):
        n = 30
        df = pd.DataFrame(
            {
                "department": ["PSYCH"] * n,
                "cluster_id": [0] * n,
                "diversity_score": np.random.rand(n),
                "raw_text": [f"text {i}" for i in range(n)],
                "language": ["en"] * n,
            }
        )
        selected, report = undersample_department(df, target_size=50)
        assert len(selected) == 30
        assert report["action"] == "passthrough"
