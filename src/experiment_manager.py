"""Experiment manager for tracking and dataset validation."""

import hashlib
import json
import platform
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
import torch

from src.config_manager import TrainingConfig


def get_git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return "N/A"


class ExperimentManager:
    def __init__(
        self,
        config: TrainingConfig,
        dataset_path: str = "meditriage/data/processed/dataset.parquet",
    ):
        self.config = config
        self.dataset_path = Path(dataset_path)

        # Generate new experiment ID if not set
        self.experiment_id = str(uuid.uuid4())

        self.results_dir = Path(config.checkpoint_dir) / self.experiment_id
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def generate_dataset_manifest(self) -> dict:
        """Generates the dataset manifest required for strict validation."""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at {self.dataset_path}")

        pq_file = pq.ParquetFile(self.dataset_path)

        total_rows = 0
        splits = {"train": 0, "val": 0, "test": 0}
        departments = {}
        severities = {}

        # We also need a fast fingerprint.
        # For a 7.6M row dataset, hashing the entire file can take minutes.
        # We will hash the first 1MB of the file and its size as a proxy fingerprint.
        file_size = self.dataset_path.stat().st_size
        with open(self.dataset_path, "rb") as f:
            head_bytes = f.read(1024 * 1024)
        fingerprint_hash = hashlib.sha256(
            head_bytes + str(file_size).encode()
        ).hexdigest()

        for i in range(pq_file.num_row_groups):
            df = pq_file.read_row_group(
                i, columns=["split", "department", "triage_level"]
            ).to_pandas()
            total_rows += len(df)

            for s, c in df["split"].value_counts().items():
                splits[s] = splits.get(s, 0) + c

            for d, c in df["department"].value_counts(dropna=False).items():
                d_key = str(d)
                departments[d_key] = departments.get(d_key, 0) + c

            for t, c in df["triage_level"].value_counts(dropna=False).items():
                t_key = str(t)
                severities[t_key] = severities.get(t_key, 0) + c

        manifest = {
            "dataset_fingerprint": fingerprint_hash,
            "total_rows": total_rows,
            "train_rows": splits.get("train", 0),
            "validation_rows": splits.get("val", 0),
            "test_rows": splits.get("test", 0),
            "department_distribution": departments,
            "severity_distribution": severities,
            "builder_commit": get_git_commit(),
            "dataset_build_timestamp": datetime.fromtimestamp(
                self.dataset_path.stat().st_mtime, tz=timezone.utc
            ).isoformat(),
        }

        return manifest

    def validate_or_create_manifest(self) -> str:
        """Validates existing manifest or creates a new one, returning the manifest hash."""
        manifest_path = self.results_dir / "dataset_manifest.json"
        current_manifest = self.generate_dataset_manifest()

        if manifest_path.exists():
            with open(manifest_path, "r") as f:
                saved_manifest = json.load(f)

            if (
                current_manifest["dataset_fingerprint"]
                != saved_manifest["dataset_fingerprint"]
            ):
                raise ValueError(
                    "Dataset manifest validation failed: The dataset fingerprint has changed! Aborting to prevent corrupted resumes."
                )
            if current_manifest["total_rows"] != saved_manifest["total_rows"]:
                raise ValueError(
                    "Dataset manifest validation failed: Total rows mismatch!"
                )

        else:
            with open(manifest_path, "w") as f:
                json.dump(current_manifest, f, indent=2)

        # Return hash of manifest
        return hashlib.sha256(
            json.dumps(current_manifest, sort_keys=True).encode()
        ).hexdigest()

    def setup_experiment(self, tokenizer_version: str, model_version: str) -> dict:
        """Creates experiment tracking metadata."""
        dataset_manifest_hash = self.validate_or_create_manifest()

        gpu_info = "CPU"
        if torch.cuda.is_available():
            gpu_info = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]

        metadata = {
            "experiment_id": self.experiment_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "git_commit": get_git_commit(),
            "dataset_manifest_hash": dataset_manifest_hash,
            "config": self.config.to_dict(),
            "config_hash": self.config.get_hash(),
            "random_seed": self.config.seed,
            "hardware": {
                "gpus": gpu_info,
                "cpu_arch": platform.machine(),
                "os": platform.system(),
            },
            "versions": {
                "pytorch": torch.__version__,
                "cuda": torch.version.cuda if torch.cuda.is_available() else "N/A",
                "tokenizer": tokenizer_version,
                "model": model_version,
            },
        }

        with open(self.results_dir / "experiment_manifest.json", "w") as f:
            json.dump(metadata, f, indent=2)

        return metadata
